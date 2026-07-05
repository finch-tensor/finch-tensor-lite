from finchlite.autoschedule import (
    DefaultLoopOrderer,
    DefaultLogicFormatter,
    DefaultLogicOptimizer,
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
