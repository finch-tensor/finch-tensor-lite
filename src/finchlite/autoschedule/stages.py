from abc import abstractmethod

from finchlite import finch_einsum as ein
from finchlite.algebra import ffuncs
from finchlite.finch_logic import LogicStatement, LogicNode, Alias, Field, Aggregate, Reorder, Literal, MapJoin, Table, Produces, Plan, Query
from finchlite import finch_notation as ntn
from finchlite.algebra.tensor import TensorFType
from finchlite.finch_logic.tensor_stats import StatsFactory, TensorStats
from finchlite.symbolic import PreWalk, Rewrite, Stage

class LogicNotationLowerer(Stage):
    @abstractmethod
    def transform(
        self, term: LogicStatement, bindings: dict[Alias, TensorFType], 
            stats: dict[Alias, TensorStats], stats_factory: StatsFactory
    ) -> tuple[ntn.Module]:
        """
        Generate Finch Notation from the given logic and input types.  Also
        return a dictionary including additional tables needed to run the kernel.
        """

class LogicEinsumLowerer(Stage):
    @abstractmethod
    def transform(
        self, term: LogicStatement, bindings: dict[Alias, TensorFType], 
            stats: dict[Alias, TensorStats], stats_factory: StatsFactory
    ) -> tuple[ein.EinsumStatement, dict[ein.Alias, TensorFType]]:
        """
        Generate Einsum Notation from the given logic and input types.  Also
        return a dictionary including additional tables needed to run the kernel.
        """

class AliasedForm(Stage):
    """
    AliasedForm requires that all aliases in the input are defined in the bindings or in previous queries and
    that all Tables are wrapping Aliases.
    """
    def validate_inputs(self, term: LogicNode, bindings: dict[Alias, TensorFType], 
                        stats: dict[Alias, TensorStats], stats_factory: StatsFactory) -> None:
        defined_aliases = set(bindings.keys())
        def validate(node):
            match node:
                case Query(Alias() as lhs, _):
                    defined_aliases.add(lhs)
                case Alias(name):
                    if node not in bindings:
                        raise ValueError(f"Alias {name} is not defined in bindings.")
                case Table(tns, idxs):
                    if not isinstance(tns, Alias):
                        raise ValueError("Table nodes must wrap an Alias.")
            return node
        Rewrite(PreWalk(validate))(term)   

class SingleAggregateForm(AliasedForm):
    """
    SingleAggregateForm assumes that the fusion strategy has already been optimized for this query. In particular,
    they allow four valid kinds of input query:
    1) transpose queries Query(_, Reorder(Table(), _))
    2) aggregate queries w/out an output order Query(_, Aggregate(_, _, arg, _))
    3) aggregate queries with an output order Query(_, Reorder(_, Aggregate(_, _, arg, _)), output_order)
    4) in-place queries Query(lhs, Reorder(MapJoin(op1, (Table(lhs, output_order), Aggregate(op2, _, arg, _)), _), output_order) 
    (Here, op2 can be ffunc.overwrite or it can be equal to op1).
    """
    def validate_inputs(self, term: Plan, bindings: dict[Alias, TensorFType], 
                        stats: dict[Alias, TensorStats], stats_factory: StatsFactory) -> None:
        super.validate_inputs(term, bindings, stats, stats_factory)
        def validate(node, agg_allowed):
            match node:
                case Plan(bodies):
                    if not isinstance(bodies[-1], Produces):
                        raise ValueError("The last body of a plan must be a Produces node.")
                    for body in bodies[:-1]:
                        validate(body, True)
                case Query(Alias(), Reorder(_, Aggregate(_, _, arg, _))):
                    return validate(arg, False)
                case Query(Alias(), Aggregate(_, _, arg, _)):
                    return validate(arg, False)
                case Query(Alias() as lhs1, Reorder(MapJoin(op1, (Table(lhs2, output_order1), Aggregate(op2, _, arg, _))), output_order2)):
                    if lhs1 != lhs2:
                        raise ValueError("In-place queries must have the same alias on the left-hand side and inside the MapJoin.")
                    if output_order1 != output_order2:
                        raise ValueError("In-place queries must read and write in the same order.")
                    if op2 not in (ffuncs.overwrite, op1):
                        raise ValueError("The aggregate operator in an in-place query must be either ffunc.overwrite or the same as the MapJoin operator.")
                    return validate(arg, False)
                case Aggregate(_, _, arg, _):
                    if not agg_allowed:
                        raise ValueError("Nested aggregates are not supported.")
                    return validate(arg, False)
                case Reorder(arg, _):
                    return validate(arg, agg_allowed)
                case Literal() | Alias() | Table():
                    return None
                case _:
                    raise ValueError(f"Unsupported query type: {node}")
        validate(term, True)

class LoopOrderedForm(SingleAggregateForm):
    """
    LoopOrderedForm assumes that the input query has had its loop order set. There are four valid
    forms for a query in LoopOrderedForm:
        1) transpose queries Query(_, Reorder(Table(), _))
        2) aggregate queries w/out an output order Query(_, Aggregate(_, _, Reorder(arg, loop_order), _))
        3) aggregate queries with an output order Query(_, Reorder(_, Aggregate(_, _, Reorder(arg, loop_order), _)), output_order)
        4) in-place queries Query(lhs, Reorder(MapJoin(_, (Table(lhs, lhs_idxs), Aggregate(_,_, Reorder(agg_arg, loop_order), _))), lhs_idxs)))
    """
    @staticmethod
    def _check_loop_order(idxs, loop_order): 
        rel_loop_order = [idx for idx in loop_order if idx in idxs]
        return rel_loop_order == idxs

    def validate_inputs(self, term: Plan, bindings: dict[Alias, TensorFType], 
                        stats: dict[Alias, TensorStats], stats_factory: StatsFactory) -> None:
        super.validate_inputs(term, bindings, stats, stats_factory)
        def validate(node, loop_order):
            match node:
                case Plan(bodies):
                    for body in bodies[:-1]:
                        validate(body, loop_order)
                case Query(Alias(), Reorder(Table(tns, idxs), loop_order)):
                    return
                case Query(Alias(), Aggregate(_, _, Reorder(arg, loop_order), _)):
                    return validate(arg, loop_order)
                case Query(Alias(), Reorder(Aggregate(_, _, Reorder(arg, loop_order)), _)):
                    return validate(arg, loop_order)
                case Query(Alias(), Aggregate(_, _, arg, _)):
                    raise ValueError("All aggregates must wrap a Reorder node specifying the loop order.")
                case Query(Alias(), Reorder(MapJoin(_, (Table(lhs, lhs_idxs), Aggregate(_, _, Reorder(agg_arg, idxs_1), _))), _)):
                    if not lhs_idxs == idxs_1[:len(lhs_idxs)]:
                        raise ValueError("The output idxs of an in-place query must be a prefix of the loop order.")
                    return validate(agg_arg, idxs_1)
                case Query(Alias(), Reorder(MapJoin(_, (Table(), Aggregate(_, _, arg, _))), _)):
                    raise ValueError("In-place queries must have an interior loop order!")
                case MapJoin(_, args):
                    for arg in args:
                        validate(arg)
                case Table(tns, idxs):
                    if not self._check_loop_order(idxs, loop_order):
                        raise ValueError("Table index order does not match loop order.")
                case Reorder(arg, _):
                    raise ValueError("Reorder nodes should only appear in transposes, output orders, and loop orders!")
                case Literal():
                    return
                case _:
                    raise ValueError(f"Unsupported query type: {node}")
        validate(term)

class FormattedForm(LoopOrderedForm):
    """
    FormattedForm requires that the input query has had its tensor formats and output orders set. There are three valid
    forms for a query in FormattedForm:
        1) transpose queries Query(_, Reorder(Table(), _))
        2) aggregate queries with an output and loop order Query(_, Reorder(_, Aggregate(_, _, Reorder(arg, loop_order), _)), output_order)
        3) in-place queries Query(lhs, Reorder(MapJoin(_, (Table(lhs, lhs_idxs), Aggregate(_,_, Reorder(agg_arg, loop_order), _))), lhs_idxs)))
    """
    def validate_inputs(self, term: Plan, bindings: dict[Alias, TensorFType], 
                        stats: dict[Alias, TensorStats], stats_factory: StatsFactory) -> None:
        super.validate_inputs(term, bindings, stats, stats_factory)
        def validate(node):
            match node:
                case Plan(bodies):
                    for body in bodies[:-1]:
                        validate(body)
                case Query(Alias(), Reorder(Table(tns, idxs), loop_order)):
                    return
                case Query(Alias(), Reorder(Aggregate(_, _, arg, _),_)):
                    return validate(arg)
                case Query(Alias(), Reorder(MapJoin(_, (Table(lhs, lhs_idxs), Aggregate(_, _, Reorder(agg_arg, idxs_1), _))), _)):
                    if not lhs_idxs == idxs_1[:len(lhs_idxs)]:
                        raise ValueError("The output idxs of an in-place query must be a prefix of the loop order.")
                    return validate(agg_arg, idxs_1)
                case Query(Alias(), Aggregate(_, _, arg, _)):
                    raise ValueError("All aggregates must be wrapped in a Reorder node specifying the output order.")
                case MapJoin(_, args):
                    for arg in args:
                        validate(arg)
                case Table(tns, _):
                    if tns not in bindings:
                        raise ValueError(f"Alias {tns.name} is not defined in bindings. All aliases must have TensorFTypes specified at this stage.")
                case Literal():
                    return
                case _:
                    raise ValueError(f"Unsupported query type: {node}")
        validate(term)