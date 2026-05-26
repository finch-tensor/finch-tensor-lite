import logging
from abc import abstractmethod
from functools import reduce
from itertools import chain as join_chains
from typing import Literal

from finchlite.algebra import TensorFType, ffuncs
from finchlite.algebra.utils import intersect
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    LogicExpression,
    LogicLoader,
    LogicNode,
    LogicStatement,
    Literal as LogicLiteral,
    MapJoin,
    MockLogicLoader,
    Plan,
    Produces,
    Query,
    Reorder,
    StatsFactory,
    Table,
    TensorStats,
)
from finchlite.symbolic import Namespace, PostOrderDFS, PostWalk, PreWalk, Rewrite
from finchlite.util.logging import LOG_LOGIC_POST_OPT

from .standardize import concordize

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


def _align(
    ex: LogicExpression, loop_order: tuple[Field, ...]
) -> tuple[LogicExpression, tuple[Query, ...]]:
    """Align each ``Table`` / ``Reorder(Table, ...)`` to ``loop_order``."""
    needed_swizzles: dict[
        tuple[Alias, tuple[Field, ...], tuple[Field, ...]], Alias
    ] = {}
    namespace = Namespace(ex)

    def rule(node: LogicNode) -> LogicNode | None:
        match node:
            case Reorder(Table(Alias() as var, physical), logical):
                pass
            case Table(Alias() as var, physical):
                logical = physical
            case _:
                return None

        field_set = set(logical)
        desired = tuple(f for f in loop_order if f in field_set)
        if desired == logical:
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


def _galley_standardize(
    root: LogicStatement, bindings: dict[Alias, TensorFType]
) -> LogicStatement:
    """Normalize Galley query roots into loop-orderer input grammar."""
    fill_values = None

    def get_fill_values():
        nonlocal fill_values
        if fill_values is None:
            fill_values = root.infer_fill_value(
                {var: val.fill_value for var, val in bindings.items()}
            )
        return fill_values

    def rule(node: LogicNode) -> LogicNode | None:
        match node:
            case Query(
                lhs,
                Reorder(Aggregate(op, init, arg, reduce_axes), output_order),
            ):
                loop_order = output_order + tuple(
                    idx for idx in reduce_axes if idx not in output_order
                )
                return Query(
                    lhs,
                    Aggregate(op, init, Reorder(arg, loop_order), reduce_axes),
                )
            case Query(lhs, Aggregate(op, init, arg, reduce_axes)) if not isinstance(
                arg, Reorder
            ):
                output_order = tuple(arg.fields())
                loop_order = output_order + tuple(
                    idx for idx in reduce_axes if idx not in output_order
                )
                return Query(
                    lhs,
                    Aggregate(op, init, Reorder(arg, loop_order), reduce_axes),
                )
            case Query(lhs, Aggregate()):
                return None
            case Query(lhs, Reorder(Table(Alias(), _), _)):
                return None
            case Query(lhs, rhs):
                return Query(
                    lhs,
                    Aggregate(
                        LogicLiteral(ffuncs.overwrite),
                        LogicLiteral(rhs.fill_value(get_fill_values())),
                        Reorder(rhs, rhs.fields()),
                        (),
                    ),
                )
        return None

    return Rewrite(PostWalk(rule))(root)


def _check_loop_order(
    idxs: tuple[Field, ...],
    loop_order: tuple[Field, ...],
) -> None:
    field_set = set(idxs)
    desired = tuple(f for f in loop_order if f in field_set)
    if desired != idxs:
        raise ValueError("Table indices do not match loop order")


# Validation func
def _validate_input_query(query: Query) -> None:
    """
    Validate that a Query rhs matches the loop-ordering grammar:
    ``Aggregate`` (inner form unrestricted),
    or ``Reorder`` of a ``Table``), at most
    one ``Aggregate``,
    ``MapJoin`` only under an ``Aggregate`` body.

    Used by :func:`validate`.
    """
    prefix = "Invalid loop ordering input:"

    def walk(ex: LogicNode, *, at_root: bool = False) -> None:
        if at_root:
            match ex:
                case Aggregate(_, _, _, _):
                    pass
                case Reorder(Table(_, _), _):
                    pass
                case _:
                    raise ValueError(
                        f"{prefix} Query rhs must be Reorder(...) or Aggregate(...)"
                    )

        n_agg = 0

        def rule(node: LogicNode) -> LogicNode | None:
            nonlocal n_agg
            match node:
                case Aggregate(_, _, _, _):
                    n_agg += 1
                case MapJoin():
                    if n_agg == 0:
                        raise ValueError(
                            "Invalid loop ordering: MapJoin is only allowed "
                            "inside an Aggregate argument"
                        )
                case _:
                    return None
            return None

        Rewrite(PreWalk(rule))(ex)
        if n_agg > 1:
            raise ValueError(
                "Invalid loop ordering: at most one Aggregate per Query rhs"
            )

    walk(query.rhs, at_root=True)


def _validate_output_query(query: Query) -> None:
    prefix = "Invalid loop ordering output:"

    def walk(ex: LogicNode, *, at_root: bool = False) -> None:
        loop_order: tuple[Field, ...] | None = None
        if at_root:
            match ex:
                case Aggregate(_, _, Reorder(_, order), _):
                    loop_order = order
                case Reorder(Table(_, idxs), order):
                    if idxs == order:
                        loop_order = order
                        _check_loop_order(idxs, order)
                case _:
                    raise ValueError(
                        f"{prefix} Query rhs must be "
                        "Reorder(...) or Aggregate(..., Reorder(...), ...)"
                    )

        n_agg = 0

        def rule(node: LogicNode) -> LogicNode | None:
            nonlocal n_agg
            match node:
                case Aggregate(_, _, Reorder(_, _), _):
                    n_agg += 1
                case Aggregate():
                    n_agg += 1
                case Table(_, idxs) if loop_order is not None:
                    _check_loop_order(idxs, loop_order)
                case Reorder(Table(_, _), idxs) if loop_order is not None:
                    _check_loop_order(idxs, loop_order)
                case MapJoin():
                    if n_agg == 0:
                        raise ValueError(
                            "Invalid loop ordering: MapJoin is only allowed "
                            "inside an Aggregate argument"
                        )
                case _:
                    return None
            return None

        Rewrite(PreWalk(rule))(ex)
        if n_agg > 1:
            raise ValueError(
                "Invalid loop ordering: at most one Aggregate per Query rhs"
            )

    walk(query.rhs, at_root=True)


def validate(
    prgm: LogicStatement,
    *,
    kind: Literal["input", "output"] = "input",
) -> None:
    """Reject programs outside the loop-ordering grammar."""
    prefix = f"Invalid loop ordering {kind}:"
    validate_query = (
        _validate_input_query if kind == "input" else _validate_output_query
    )
    match prgm:
        case Plan(bodies):
            seen_produces = False
            for i, body in enumerate(bodies):
                if seen_produces:
                    raise ValueError(f"{prefix} Produces must be final body")
                match body:
                    case Query() as query:
                        validate_query(query)
                    case Produces(_):
                        seen_produces = True
                        if i != len(bodies) - 1:
                            raise ValueError(f"{prefix} Produces must be final body")
                    case _:
                        raise ValueError(
                            f"{prefix} expected Query or Produces in Plan, "
                            f"got {type(body).__name__}"
                        )
        case Query() as query:
            validate_query(query)
        case _:
            raise ValueError(
                f"{prefix} expected Plan or Query, got {type(prgm).__name__}"
            )


# End validation funcs


class LoopOrderer(LogicLoader):
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

    def __call__(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        #prgm = _galley_standardize(prgm, bindings)
        validate(prgm, kind="input")

        def apply_loop_order(node: LogicStatement) -> LogicStatement:
            match node:
                case Plan(bodies):
                    return Plan(tuple(apply_loop_order(body) for body in bodies))
                case Query(lhs, rhs):
                    loop_order = self.get_loop_order(
                        node, bindings, stats, stats_factory
                    )
                    match rhs:
                        # This case is used to handle the case where the loop order 
                        # is already set by the previous loop ordering step.
                        case Aggregate(_, _, Reorder(_, old_loop_order), _) if bindings:
                            loop_order = old_loop_order
                    rhs, swizzles = _align(rhs, loop_order)
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
                        case Reorder(Table(_, _), _old):
                            ordered = node
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
        prgm = concordize(prgm, bindings)
        validate(prgm, kind="output")
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
