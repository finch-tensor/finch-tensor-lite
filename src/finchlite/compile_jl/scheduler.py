from finchlite.autoschedule import (
    DefaultLogicFormatter,
    DefaultLogicOptimizer,
    DefaultLoopOrderer,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
    LogicStandardizer,
)

from .compiler import FinchJLCompiler

COMPILE_JULIA = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            LogicStandardizer(
                DefaultLoopOrderer(
                    DefaultLogicFormatter(LogicCompiler(FinchJLCompiler()))
                )
            )
        )
    )
)
