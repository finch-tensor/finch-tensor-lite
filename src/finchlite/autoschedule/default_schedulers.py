import threading
from finchlite.autoschedule.optimize import DefaultLogicOptimizer
from finchlite.codegen import NumbaCompiler
from finchlite.compile import NotationCompiler
from finchlite.finch_assembly import AssemblyInterpreter, AssemblySimplify
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicInterpreter,
    MapJoin,
    MockLogicLoader,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    Table,
    Value,
)
from finchlite.finch_logic.stages import LogicEvaluator
from finchlite.finch_notation.interpreter import NotationInterpreter
from finchlite.symbolic import PostOrderDFS, PostWalk, PreWalk

from .compiler import LogicCompiler, NotationGenerator
from .executor import LogicExecutor
from .formatter import DefaultLogicFormatter, LogicFormatter
from .galley_optimize import GalleyLogicalOptimizer
from .normalize import LogicNormalizer, normalize_names
from .stages import LogicEinsumLowerer, LogicNotationLowerer
from .standardize import LogicStandardizer
from contextlib import contextmanager

INTERPRET_LOGIC = LogicInterpreter()
OPTIMIZE_LOGIC = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            LogicStandardizer(DefaultLogicFormatter(MockLogicLoader()))
        )
    )
)
INTERPRET_NOTATION = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            LogicStandardizer(
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
            )
        )
    )
)
INTERPRET_ASSEMBLY = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            LogicStandardizer(
                DefaultLogicFormatter(
                    LogicCompiler(NotationCompiler(AssemblyInterpreter()))
                )
            )
        )
    )
)
COMPILE_NUMBA = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            LogicStandardizer(
                DefaultLogicFormatter(
                    LogicCompiler(
                        NotationCompiler(
                            NumbaCompiler(), ctx_transforms=(AssemblySimplify(),)
                        )
                    )
                )
            )
        )
    )
)

INTERPRET_NOTATION_GALLEY = LogicNormalizer(
    LogicExecutor(
        GalleyLogicalOptimizer(
            LogicStandardizer(
                DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
            )
        )
    )
)

_DEFAULT_SCHEDULER = threading.local()

def set_default_scheduler(ctx: LogicEvaluator):
    if ctx is not None:
        _DEFAULT_SCHEDULER.value = ctx

def get_default_scheduler() -> LogicEvaluator:
    try:
        return _DEFAULT_SCHEDULER.value
    except AttributeError:
        return INTERPRET_NOTATION

@contextmanager
def with_default_scheduler(ctx: LogicEvaluator):
    old_value = getattr(_DEFAULT_SCHEDULER, 'value', None)
    _DEFAULT_SCHEDULER.value = ctx
    try:
        yield
    finally:
        if old_value is None:
            if hasattr(_DEFAULT_SCHEDULER, 'value'):
                delattr(_DEFAULT_SCHEDULER, 'value')
        else:
            _DEFAULT_SCHEDULER.value = old_value