import logging
from abc import abstractmethod
from collections import Counter

from ..algebra import TensorFType
from ..finch_logic import (
    Aggregate,
    Alias,
    Field,
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
from ..util.logging import LOG_LOGIC_POST_OPT
from .tensor_stats import DenseStatsFactory

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


def _is_standardized_query(node: Query) -> bool:
    match node:
        case Query(_, Reorder(_, _)):
            return True
        case Query(_, Aggregate(_, _, Reorder(_, _), _)):
            return True
        case _:
            return False


def _assert_one_aggregrate(rhs: LogicNode) -> None:
    num_aggregrates = 0
    stack = [rhs]
    while stack:
        ex = stack.pop()
        match ex:
            case Aggregate(_, _, arg, _):
                num_aggregrates += 1
                stack.append(arg)
            case Reorder(arg, _):
                stack.append(arg)
            case Table(_, _):
                pass
            case _:
                if isinstance(ex, LogicTree):
                    stack.extend(ex.children)
    if num_aggregrates > 1:
        raise ValueError("Invalid loop ordering: at most one Aggregate per Query rhs")


def _assert_mapjoin_inside_aggregate(rhs: LogicNode) -> None:
    stack: list[tuple[LogicNode, bool]] = [(rhs, False)]
    while stack:
        ex, inside = stack.pop()
        match ex:
            case MapJoin():
                if not inside:
                    raise ValueError(
                        "Invalid loop ordering: MapJoin is only allowed "
                        "inside an Aggregate argument"
                    )
            case Aggregate(_, _, arg, _):
                stack.append((arg, True))
                continue
            case Reorder(arg, _):
                stack.append((arg, inside))
                continue
            case Table(_, _):
                continue
            case _:
                pass
        if isinstance(ex, LogicTree):
            stack.extend((child, inside) for child in ex.children)


def _validate_query(query: Query, *, kind: str) -> None:
    """
    Validate that a Query rhs matches the loop-ordering grammar: standardized
    shape, single Aggregate, MapJoin placement, no outer Reorder around
    Aggregate.

    Used for both ``validate_input`` and ``validate_output``.
    """
    prefix = f"Invalid loop ordering {kind}:"
    if not _is_standardized_query(query):
        raise ValueError(
            f"{prefix} Query rhs must be "
            "Reorder(...) or Aggregate(..., Reorder(...), ...)"
        )
    match query:
        case Query(_, Aggregate() as rhs):
            _assert_one_aggregrate(rhs)
            _assert_mapjoin_inside_aggregate(rhs)
        case Query(_, Reorder(inner, _)) if not isinstance(inner, Aggregate):
            _assert_one_aggregrate(inner)
            _assert_mapjoin_inside_aggregate(inner)
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
                        case _:
                            return Query(lhs, Reorder(rhs, loop_order))
                case Produces(_):
                    return node
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for loop ordering: {node}"
                    )

        prgm = reorder(prgm)
        validate_output(prgm)
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
