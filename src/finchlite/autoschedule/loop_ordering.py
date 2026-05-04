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
    MockLogicLoader,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)
from ..util.logging import LOG_LOGIC_POST_OPT

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
        raise ValueError(
            "Invalid loop ordering input: at most one Aggregate per Query rhs"
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
                        if not _is_standardized_query(query):
                            raise ValueError(
                                "Invalid loop ordering input: expected "
                                "standardized Query rhs"
                            )
                        match query:
                            case Query(_, rhs):
                                _assert_one_aggregrate(rhs)
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
            if not _is_standardized_query(query):
                raise ValueError(
                    "Invalid loop ordering input: expected standardized Query rhs"
                )
            match query:
                case Query(_, rhs):
                    _assert_one_aggregrate(rhs)
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
                    case Query(_, Reorder(_, _)):
                        pass
                    case Query():
                        raise ValueError(
                            "Invalid loop ordering output: Query rhs must be Reorder"
                        )
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
        case Query(_, Reorder(_, _)):
            pass
        case Query():
            raise ValueError("Invalid loop ordering output: Query rhs must be Reorder")
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
    ):
        validate_input(prgm)

        def reorder(node: LogicStatement) -> LogicStatement:
            match node:
                case Plan(bodies):
                    return Plan(tuple(reorder(body) for body in bodies))
                case Query(lhs, rhs):
                    loop_order = self.get_loop_order(node, bindings)
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
        return self.loader(prgm, bindings)


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

        output_fields: tuple[Field, ...] | None = None
        match node:
            case Query(_, Aggregate() as aggregate):
                visit(aggregate)
                output_fields = tuple(aggregate.fields())
            case Query(_, rhs):
                visit(rhs)

        if output_fields is not None:
            ordered = sorted(
                output_fields,
                key=lambda idx: occurrences[idx],
                reverse=True,
            )
        else:
            ordered = sorted(
                occurrences.keys(),
                key=lambda idx: occurrences[idx],
                reverse=True,
            )
        return tuple(ordered)
