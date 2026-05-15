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
from finchlite.finch_notation.interpreter import NotationInterpreter
from finchlite.symbolic import PostOrderDFS, PostWalk, PreWalk

from .compiler import LogicCompiler, NotationGenerator
from .executor import LogicExecutor
from .formatter import DefaultLogicFormatter, LogicFormatter
from .galley_optimize import GalleyLogicalOptimizer
from .normalize import LogicNormalizer, normalize_names
from .stages import LogicEinsumLowerer, LogicNotationLowerer
from .standardize import LogicStandardizer

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

__all__ = [
    "COMPILE_NUMBA",
    "INTERPRET_ASSEMBLY",
    "INTERPRET_LOGIC",
    "INTERPRET_NOTATION",
    "INTERPRET_NOTATION_GALLEY",
    "OPTIMIZE_LOGIC",
    "Aggregate",
    "Alias",
    "DefaultLogicFormatter",
    "DefaultLogicOptimizer",
    "Field",
    "Literal",
    "LogicCompiler",
    "LogicEinsumLowerer",
    "LogicExecutor",
    "LogicFormatter",
    "LogicNormalizer",
    "LogicNotationLowerer",
    "LogicStandardizer",
    "MapJoin",
    "NotationGenerator",
    "Plan",
    "PostOrderDFS",
    "PostWalk",
    "PreWalk",
    "Produces",
    "Query",
    "Relabel",
    "Reorder",
    "Table",
    "Value",
    "normalize_names",
]
