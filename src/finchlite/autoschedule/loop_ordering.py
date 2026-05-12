import logging
from abc import abstractmethod
from collections import Counter
from typing import cast

from ..algebra import TensorFType
from ..finch_logic import (
    Aggregate,
    Alias,
    Field,
    LogicExpression,
    LogicLoader,
    LogicNode,
    LogicStatement,
    LogicTree,
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
from ..symbolic import PostWalk, Rewrite
from ..util.logging import LOG_LOGIC_POST_OPT
from .normalize import normalize_names
from .standardize import concordize, drop_reorders, flatten_plans
from .tensor_stats import DenseStatsFactory

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


# Tranpose funcs
def _get_operand_table_and_idxs(
    arg: LogicExpression,
) -> tuple[Table, tuple[Field, ...]] | None:
    match arg:
        case Table(_, _) as t:
            return (t, t.idxs)
        case Reorder(Table(_, _) as t, idxs):
            return (t, idxs)
        case _:
            return None


def _transpose(t: Table, reordered: tuple[Field, ...]) -> Reorder:
    return Reorder(Table(t.tns, t.idxs), reordered)


def _transpose_tables(mj: MapJoin, loop_order: tuple[Field, ...]) -> MapJoin:
    views = tuple(_get_operand_table_and_idxs(a) for a in mj.args)
    if any(v is None for v in views):
        return mj
    table_views = cast(tuple[tuple[Table, tuple[Field, ...]], ...], views)
    seqs = tuple(v[1] for v in table_views)
    union: set[Field] = set()
    for seq in seqs:
        union.update(seq)
    canonical = tuple(f for f in loop_order if f in union)
    new_args: list[LogicExpression] = []
    changed = False
    for arg, (base_t, logical) in zip(mj.args, table_views, strict=True):
        field_set = set(logical)
        new_order = tuple(f for f in canonical if f in field_set)
        if new_order != logical:
            changed = True
            new_args.append(_transpose(base_t, new_order))
        else:
            new_args.append(arg)
    if not changed:
        return mj
    return MapJoin(mj.op, tuple(new_args))


def _align(ex: LogicExpression, loop_order: tuple[Field, ...]) -> LogicExpression:
    def rule(node: LogicNode) -> LogicNode | None:
        match node:
            case MapJoin() as mj:
                aligned = _transpose_tables(mj, loop_order)
                return aligned if aligned is not mj else None
            case _:
                return None

    return Rewrite(PostWalk(rule))(ex)


# Validation func
def _contains_aggregate_or_mapjoin(ex: LogicNode) -> bool:
    stack: list[LogicNode] = [ex]
    while stack:
        curr = stack.pop()
        match curr:
            case Aggregate():
                return True
            case MapJoin():
                return True
            case _:
                if isinstance(curr, LogicTree):
                    stack.extend(curr.children)
    return False


def _validate_query(query: Query, *, kind: str) -> None:
    """
    Validate that a Query rhs matches the loop-ordering grammar:
    ``Aggregate`` with inner ``Reorder``,
    or ``Reorder`` whose inner is not an ``Aggregate``), at most
    one ``Aggregate``,
    ``MapJoin`` only under an ``Aggregate`` body.

    Used for both ``validate_input`` and ``validate_output``.
    """
    prefix = f"Invalid loop ordering {kind}:"

    def walk(ex: LogicNode, inside_aggregate: bool) -> int:
        match ex:
            case Aggregate(_, _, arg, _):
                return 1 + walk(arg, True)
            case MapJoin():
                if not inside_aggregate:
                    raise ValueError(
                        "Invalid loop ordering: MapJoin is only allowed "
                        "inside an Aggregate argument"
                    )
                n = 0
                if isinstance(ex, LogicTree):
                    for c in ex.children:
                        n += walk(c, inside_aggregate)
                return n
            case Reorder(arg, _):
                return walk(arg, inside_aggregate)
            case Table(_, _):
                return 0
            case _:
                # anything else
                n = 0
                if isinstance(ex, LogicTree):
                    for c in ex.children:
                        n += walk(c, inside_aggregate)
                return n

    match query:
        case Query(_, Aggregate(_, _, Reorder(_, _), _) as rhs):
            if walk(rhs, False) != 1:
                raise ValueError(
                    "Invalid loop ordering: at most one Aggregate per Query rhs"
                )
        case Query(_, Reorder(inner, _)) if not isinstance(inner, Aggregate):
            if walk(query.rhs, False) > 1:
                raise ValueError(
                    "Invalid loop ordering: at most one Aggregate per Query rhs"
                )
        case _:
            raise ValueError(
                f"{prefix} Query rhs must be "
                "Reorder(...) or Aggregate(..., Reorder(...), ...)"
            )


def validate_input(prgm: LogicStatement) -> None:
    match prgm:
        case Plan(bodies):
            seen_produces = False
            for i, body in enumerate(bodies):
                if seen_produces:
                    raise ValueError(
                        "Invalid loop ordering input: Produces must be final body"
                    )
                match body:
                    case Query() as query:
                        _validate_query(query, kind="input")
                    case Produces(_):
                        seen_produces = True
                        if i != len(bodies) - 1:
                            raise ValueError(
                                "Invalid loop ordering input: Produces must be "
                                "final body"
                            )
                    case _:
                        raise ValueError(
                            "Invalid loop ordering input: expected Query or "
                            f"Produces in Plan, got {type(body).__name__}"
                        )
        case Query() as query:
            _validate_query(query, kind="input")
        case _:
            raise ValueError(
                "Invalid loop ordering input: expected Plan or Query, got "
                f"{type(prgm).__name__}"
            )


def validate_output(prgm: LogicStatement) -> None:
    match prgm:
        case Plan(bodies):
            seen_produces = False
            for i, body in enumerate(bodies):
                if seen_produces:
                    raise ValueError(
                        "Invalid loop ordering output: Produces must be final body"
                    )
                match body:
                    case Query() as query:
                        _validate_query(query, kind="output")
                    case Produces(_):
                        seen_produces = True
                        if i != len(bodies) - 1:
                            raise ValueError(
                                "Invalid loop ordering output: Produces must be "
                                "final body"
                            )
                    case _:
                        raise ValueError(
                            "Invalid loop ordering output: expected Query or "
                            f"Produces in Plan, got {type(body).__name__}"
                        )
        case Query() as query:
            _validate_query(query, kind="output")
        case _:
            raise ValueError(
                "Invalid loop ordering output: expected Plan or Query, got "
                f"{type(prgm).__name__}"
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
    ) -> tuple[Field, ...]:
        """
        Return the desired loop order.
        """
        ...

    def __call__(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats] | None = None,
        stats_factory: StatsFactory | None = None,
    ):
        validate_input(prgm)
        # NOTE: change when we have a proper stats factory
        if stats is None:
            stats = {}
        if stats_factory is None:
            stats_factory = DenseStatsFactory()
        # End  NOTE

        def reorder(node: LogicStatement) -> LogicStatement:
            match node:
                case Plan(bodies):
                    return Plan(tuple(reorder(body) for body in bodies))
                case Query(lhs, rhs):
                    loop_order = self.get_loop_order(node, bindings)
                    rhs = _align(rhs, loop_order)
                    match rhs:
                        case Aggregate(
                            op,
                            init,
                            Reorder(inner, _old_loop_order),
                            reduce_axes,
                        ):
                            return Query(
                                lhs,
                                Aggregate(
                                    op,
                                    init,
                                    Reorder(inner, loop_order),
                                    reduce_axes,
                                ),
                            )
                        case Reorder(inner, _old) if not _contains_aggregate_or_mapjoin(
                            inner
                        ):
                            return node
                        case _:
                            return Query(lhs, Reorder(rhs, loop_order))
                case Produces(_):
                    return node
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for loop ordering: {node}"
                    )

        prgm = reorder(prgm)
        # mutate bindings_out
        bindings_out = dict(bindings)
        prgm = concordize(prgm, bindings_out)
        validate_output(prgm)
        prgm = drop_reorders(prgm)
        prgm = flatten_plans(prgm)
        prgm, bindings_out = normalize_names(prgm, bindings_out)
        validate_output(prgm)
        logger.debug(prgm)
        return self.loader(prgm, bindings_out, stats, stats_factory)


class DefaultLoopOrderer(LoopOrderer):
    """
    A simple occurrence-based loop-ordering.

    Referenced from
    ``PhysicalOptimizer/loop-ordering.jl``.
    """

    def __init__(self, loader: LogicLoader | None = None):
        super().__init__(loader)

    def get_loop_order(
        self,
        node: Query,
        bindings: dict[Alias, TensorFType],
    ) -> tuple[Field, ...]:
        """
        Count how many times each index appears across all tensors.
        Then do sorting
        """
        occurrences: Counter[Field] = Counter()

        def visit(ex) -> None:
            match ex:
                case Table(_, idxs):
                    for idx in idxs:
                        occurrences[idx] += 1
                case Aggregate(_, _, arg, _):
                    visit(arg)
                case Reorder(arg, _):
                    visit(arg)
                case _:
                    if isinstance(ex, LogicTree):
                        for child in ex.children:
                            visit(child)

        match node:
            case Query(_, rhs):
                visit(rhs)

        ordered = sorted(
            occurrences.keys(),
            key=lambda idx: occurrences[idx],
            reverse=True,
        )
        return tuple(ordered)
