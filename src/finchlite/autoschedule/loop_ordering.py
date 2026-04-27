import logging
from abc import abstractmethod
from collections import Counter

from ..algebra import TensorFType
from ..finch_logic import (
    Aggregate,
    Alias,
    Field,
    LogicLoader,
    LogicStatement,
    MockLogicLoader,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)
from ..util.logging import LOG_LOGIC_POST_OPT

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


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
        first_seen: dict[Field, int] = {}
        clock = 0

        def visit(ex) -> None:
            nonlocal clock
            match ex:
                case Table(_, idxs):
                    for idx in idxs:
                        if idx not in first_seen:
                            first_seen[idx] = clock
                            clock += 1
                        occurrences[idx] += 1
                case Aggregate(_, _, arg, _):
                    visit(arg)
                case Reorder(arg, _):
                    visit(arg)
                case _:
                    # MapJoin and other composites expose their children via
                    # standard dataclass fields; walk them generically.
                    for child in getattr(ex, "__dataclass_fields__", {}).keys():
                        val = getattr(ex, child)
                        if isinstance(val, tuple):
                            for item in val:
                                if hasattr(item, "__dataclass_fields__"):
                                    visit(item)
                        elif hasattr(val, "__dataclass_fields__"):
                            visit(val)

        match node:
            case Query(_, rhs):
                visit(rhs)

        # Sort: highest occurrence first; break ties by first-seen order.
        ordered = sorted(
            occurrences.keys(),
            key=lambda idx: (-occurrences[idx], first_seen[idx]),
        )
        return tuple(ordered)
