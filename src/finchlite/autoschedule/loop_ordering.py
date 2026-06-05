import logging
from abc import abstractmethod
from functools import reduce
from itertools import chain as join_chains

from finchlite.algebra import TensorFType
from finchlite.algebra.utils import intersect
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    LogicExpression,
    LogicLoader,
    LogicNode,
    LogicStatement,
    MockLogicLoader,
    Plan,
    Produces,
    Query,
    Reorder,
    StatsFactory,
    Table,
    TensorStats,
)
from finchlite.symbolic import Namespace, PostOrderDFS, PostWalk, Rewrite
from finchlite.util.logging import LOG_LOGIC_POST_OPT

from .stages import SingleAggregateForm
from .standardize import flatten_plans

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


def _align(
    ex: LogicExpression,
    loop_order: tuple[Field, ...],
    bindings: dict[Alias, TensorFType],
    namespace: Namespace,
) -> tuple[LogicExpression, tuple[Query, ...]]:
    """Align each ``Table`` / ``Reorder(Table, ...)`` to ``loop_order``."""
    needed_swizzles: dict[
        tuple[Alias, tuple[Field, ...], tuple[Field, ...]], Alias
    ] = {}

    def rule(node: LogicNode) -> LogicNode | None:
        match node:
            case Reorder(Table(Alias() as var, physical), logical):
                is_reorder = True
            case Table(Alias() as var, physical):
                logical = physical
                is_reorder = False
            case _:
                return None

        field_set = set(logical)
        desired = tuple(f for f in loop_order if f in field_set)
        if desired == logical:
            if is_reorder and physical == logical:
                return Table(var, physical)
            if is_reorder:
                key = (var, physical, desired)
                if key not in needed_swizzles:
                    needed_swizzles[key] = Alias(namespace.freshen(var.name))
                return Table(needed_swizzles[key], desired)
            return None
        if len(desired) != len(physical) or set(desired) != set(physical):
            return None

        key = (var, physical, desired)
        if key not in needed_swizzles:
            needed_swizzles[key] = Alias(namespace.freshen(var.name))
        return Table(needed_swizzles[key], desired)

    new_ex = Rewrite(PostWalk(rule))(ex)
    queries = tuple(
        Query(alias, Reorder(Table(var, physical), desired))
        for (var, physical, desired), alias in needed_swizzles.items()
    )
    return new_ex, queries


class LoopOrderer(SingleAggregateForm, LogicLoader):
    """
    A LoopOrderer determines the loop ordering for each query in a logic
    program. Subclasses implement ``get_loop_order`` to swap in different
    loop-ordering strategies.

    The input program is expected to be in the standardized form produced by
    ``LogicStandardizer``: every query is either a Reorder of a single
    argument, or an Aggregate over a Reorder of a series of map-joins.
    """

    def __init__(self, loader: LogicLoader | None = None):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.loader = loader

    @abstractmethod
    def get_loop_order(
        self,
        node: Query,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[Field, ...]:
        """
        Return the desired loop order.
        """
        ...

    def lower(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        namespace = Namespace(prgm)

        def output_ordered_aggregate(rhs: LogicExpression):
            # Unwrap Reorder(...)/Aggregate(...) to find an output-ordered aggregate.
            output_order = None
            while True:
                match rhs:
                    case Reorder(arg, order):
                        output_order = order
                        rhs = arg
                    case Aggregate(op, init, arg, reduce_axes) if (
                        output_order is not None
                    ):
                        return op, init, arg, reduce_axes, output_order
                    case _:
                        return None

        def strip_noop_reorders(ex: LogicExpression) -> LogicExpression:
            # Drop Reorder nodes whose order already matches the arg's fields.
            def rule(node: LogicNode) -> LogicNode | None:
                match node:
                    case Reorder(arg, order) if arg.fields() == order:
                        return arg
                    case _:
                        return None

            return Rewrite(PostWalk(rule))(ex)

        def apply_loop_order(node: LogicStatement) -> LogicStatement:
            # Rewrite each query rhs to honor the loop order, emitting swizzles.
            match node:
                case Plan(bodies):
                    return Plan(tuple(apply_loop_order(body) for body in bodies))
                case Query(lhs, rhs):
                    output_aggregate = output_ordered_aggregate(rhs)
                    if output_aggregate is not None:
                        op, init, arg, reduce_axes, output_order = output_aggregate
                        match arg:
                            # This case preserves an existing aggregate loop order
                            # from Logic Optimizer.
                            case Reorder(inner, old_loop_order):
                                arg = inner
                                loop_order = old_loop_order
                            case _:
                                loop_order = self.get_loop_order(
                                    node, bindings, stats, stats_factory
                                )
                        arg, swizzles = _align(arg, loop_order, bindings, namespace)
                        arg = strip_noop_reorders(arg)
                        ordered = Query(
                            lhs,
                            Reorder(
                                Aggregate(
                                    op,
                                    init,
                                    Reorder(arg, loop_order),
                                    reduce_axes,
                                ),
                                output_order,
                            ),
                        )
                    else:
                        match rhs:
                            case Table(_, idxs):
                                swizzles = ()
                                ordered = Query(lhs, Reorder(rhs, idxs))
                            case Reorder(arg, output_order):
                                arg, swizzles = _align(
                                    arg, output_order, bindings, namespace
                                )
                                arg = strip_noop_reorders(arg)
                                ordered = Query(lhs, Reorder(arg, output_order))
                            # This case preserves an existing aggregate loop order
                            # from Logic Optimizer.
                            case Aggregate(
                                op,
                                init,
                                Reorder(inner, old_loop_order),
                                reduce_axes,
                            ):
                                loop_order = old_loop_order
                                inner, swizzles = _align(
                                    inner, loop_order, bindings, namespace
                                )
                                inner = strip_noop_reorders(inner)
                                ordered = Query(
                                    lhs,
                                    Aggregate(
                                        op,
                                        init,
                                        Reorder(inner, loop_order),
                                        reduce_axes,
                                    ),
                                )
                            case Reorder(Table(_, _), _old):
                                swizzles = ()
                                ordered = node
                            case _:
                                loop_order = self.get_loop_order(
                                    node, bindings, stats, stats_factory
                                )
                                rhs, swizzles = _align(
                                    rhs, loop_order, bindings, namespace
                                )
                                rhs = strip_noop_reorders(rhs)
                                match rhs:
                                    case Aggregate(
                                        op,
                                        init,
                                        Reorder(inner, _old_loop_order),
                                        reduce_axes,
                                    ):
                                        ordered = Query(
                                            lhs,
                                            Aggregate(
                                                op,
                                                init,
                                                Reorder(inner, loop_order),
                                                reduce_axes,
                                            ),
                                        )
                                    case _:
                                        ordered = Query(lhs, Reorder(rhs, loop_order))
                    if swizzles:
                        return Plan((*swizzles, ordered))
                    return ordered
                case Produces(_):
                    return node
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for loop ordering: {node}"
                    )

        prgm = apply_loop_order(prgm)
        prgm = flatten_plans(prgm)
        # for mypy test, make sure prgm is a Plan
        if not isinstance(prgm, Plan):
            raise ValueError(f"Loop ordering output must be a Plan: {prgm}")
        # LoopOrderedForm.validate_inputs(prgm, bindings, stats, stats_factory)
        logger.debug(prgm)
        return self.loader(prgm, bindings, stats, stats_factory)


class DefaultLoopOrderer(LoopOrderer):
    """
    Heuristic loop ordering.

    From ``optimize._heuristic_loop_order``.
    """

    class CycleInFields(Exception): ...

    @staticmethod
    def _toposort(chains: list[list[Field]]) -> tuple[Field, ...]:
        chains = [c for c in chains if len(c) > 0]
        parents = {chain[0]: 0 for chain in chains}
        for chain in chains:
            for f in chain[1:]:
                parents[f] = parents.get(f, 0) + 1
        roots = [f for f in parents if parents[f] == 0]
        perm = []
        while len(parents) > 0:
            if len(roots) == 0:
                raise DefaultLoopOrderer.CycleInFields(
                    "Cycle detected in fields' orders"
                )
            perm.append(roots.pop())
            for chain in chains:
                if len(chain) > 0 and chain[0] == perm[-1]:
                    chain.pop(0)
                    if len(chain) > 0:
                        parents[chain[0]] -= 1
                        if parents[chain[0]] == 0:
                            roots.append(chain[0])
            parents.pop(perm[-1])
        return tuple(perm)

    @staticmethod
    def _heuristic_loop_order(root: LogicExpression) -> tuple[Field, ...]:
        chains = []
        for node in PostOrderDFS(root):
            match node:
                case Reorder(Table(_, idxs_1), idxs_2):
                    chains.append(
                        list(intersect(intersect(idxs_1, idxs_2), root.fields()))
                    )
        chains.extend([f] for f in root.fields())

        need_fix = False
        try:
            result = DefaultLoopOrderer._toposort(chains)
        except DefaultLoopOrderer.CycleInFields:
            import warnings

            warnings.warn("Cycle in fields detected, need to permute.", stacklevel=1)
            need_fix = True
            result = root.fields()

        if need_fix or reduce(max, [len(c) for c in chains], 0) < len(
            set(join_chains(*chains))
        ):
            counts: dict[Field, int] = {}
            for chain in chains:
                for f in chain:
                    counts[f] = counts.get(f, 0) + 1
            result = tuple(sorted(result, key=lambda x: counts[x] == 1))
        return result

    def __init__(self, loader: LogicLoader | None = None):
        super().__init__(loader)

    def get_loop_order(
        self,
        node: Query,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[Field, ...]:
        match node:
            case Query(_, rhs):
                return self._heuristic_loop_order(rhs)
