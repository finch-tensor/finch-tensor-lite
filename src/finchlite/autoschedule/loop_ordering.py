import logging
from abc import abstractmethod
from collections import Counter
from typing import Literal, cast

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
from ..symbolic import PostOrderDFS, PostWalk, PreWalk, Rewrite
from ..util.logging import LOG_LOGIC_POST_OPT
from .normalize import normalize_names
from .standardize import concordize, drop_reorders, flatten_plans

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


# Transpose helpers
def _get_operand_table_and_idxs(
    arg: LogicExpression,
) -> tuple[Table, tuple[Field, ...]] | None:
    """If ``arg`` is ``Table`` or ``Reorder(Table, logical)``,
    return table + idxs; else ``None``."""
    match arg:
        case Table(_, _) as t:
            return (t, t.idxs)
        case Reorder(Table(_, _) as t, idxs):
            return (t, idxs)
        case _:
            return None


def _transpose(t: Table, reordered: tuple[Field, ...]) -> Reorder:
    """Logical view ``reordered`` of the same  tensor."""
    return Reorder(Table(t.tns, t.idxs), reordered)


def _transpose_tables(mj: MapJoin, loop_order: tuple[Field, ...]) -> MapJoin:
    """Reorder each all-table ``MapJoin`` arg to match
    ``loop_order`` on shared indices."""
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
    """Line up tensor axes inside each ``MapJoin`` to ``loop_order``.

    Only touches ``Table`` operands. Loop nest order is set later in ``reorder``.
    """

    def rule(node: LogicNode) -> LogicNode | None:
        match node:
            case MapJoin() as mj:
                aligned = _transpose_tables(mj, loop_order)
                return aligned if aligned is not mj else None
            case _:
                return None

    return Rewrite(PostWalk(rule))(ex)


# Validation func
def _validate_input_query(query: Query, *, kind: str) -> None:
    """
    Validate that a Query rhs matches the loop-ordering grammar:
    ``Aggregate`` (inner form unrestricted),
    or ``Reorder`` whose inner is not an ``Aggregate``), at most
    one ``Aggregate``,
    ``MapJoin`` only under an ``Aggregate`` body.

    Used by :func:`validate`.
    """
    prefix = f"Invalid loop ordering {kind}:"

    def walk(ex: LogicNode, inside_aggregate: bool) -> None:
        in_agg = inside_aggregate

        def rule(node: LogicNode) -> LogicNode | None:
            nonlocal in_agg
            match node:
                case Aggregate(_, _, _, _):
                    in_agg = True
                case MapJoin():
                    if not in_agg:
                        raise ValueError(
                            "Invalid loop ordering: MapJoin is only allowed "
                            "inside an Aggregate argument"
                        )
                case _:
                    return None
            return None

        Rewrite(PreWalk(rule))(ex)

    match query:
        case Query(_, Aggregate(_, _, _, _)):
            pass
        case Query(_, Reorder(inner, _)) if not isinstance(inner, Aggregate):
            pass
        case _:
            raise ValueError(
                f"{prefix} Query rhs must be Reorder(...) or Aggregate(...)"
            )

    rhs = query.rhs
    n = sum(1 for node in PostOrderDFS(rhs) if isinstance(node, Aggregate))
    if n > 1:
        raise ValueError("Invalid loop ordering: at most one Aggregate per Query rhs")
    walk(rhs, False)


def _validate_output_query(query: Query, *, kind: str) -> None:
    prefix = f"Invalid loop ordering {kind}:"

    def walk(ex: LogicNode, inside_aggregate: bool) -> None:
        in_agg = inside_aggregate

        def rule(node: LogicNode) -> LogicNode | None:
            nonlocal in_agg
            match node:
                case Aggregate(_, _, Reorder(_, _), _):
                    in_agg = True
                case MapJoin():
                    if not in_agg:
                        raise ValueError(
                            "Invalid loop ordering: MapJoin is only allowed "
                            "inside an Aggregate argument"
                        )
                case _:
                    return None
            return None

        Rewrite(PreWalk(rule))(ex)

    match query:
        case Query(_, Aggregate(_, _, Reorder(_, _), _)):
            pass
        case Query(_, Reorder(inner, _)) if not isinstance(inner, Aggregate):
            pass
        case _:
            raise ValueError(
                f"{prefix} Query rhs must be "
                "Reorder(...) or Aggregate(..., Reorder(...), ...)"
            )

    rhs = query.rhs
    n = sum(1 for node in PostOrderDFS(rhs) if isinstance(node, Aggregate))
    if n > 1:
        raise ValueError("Invalid loop ordering: at most one Aggregate per Query rhs")
    walk(rhs, False)


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
                        validate_query(query, kind=kind)
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
            validate_query(query, kind=kind)
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

    After ``apply_loop_order``, runs ``concordize``, ``drop_reorders``,
    ``flatten_plans``, and ``normalize_names``
    (same as ``standardize.py`` after concordize).
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
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        validate(prgm, kind="input")

        def apply_loop_order(node: LogicStatement) -> LogicStatement:
            match node:
                case Plan(bodies):
                    return Plan(tuple(apply_loop_order(body) for body in bodies))
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
                        case Reorder(Table(_, _), _old):
                            return node
                        case _:
                            return Query(lhs, Reorder(rhs, loop_order))
                case Produces(_):
                    return node
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for loop ordering: {node}"
                    )

        prgm = apply_loop_order(prgm)
        prgm = concordize(prgm, bindings)
        validate(prgm, kind="output")
        prgm = drop_reorders(prgm)
        prgm = flatten_plans(prgm)
        prgm, bindings = normalize_names(prgm, bindings)
        validate(prgm, kind="output")
        logger.debug(prgm)
        return self.loader(prgm, bindings, stats, stats_factory)


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
            """Count index uses from ``Table`` rows (walks rhs)."""
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
