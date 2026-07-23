from finch.autoschedule import (
    DefaultLogicFormatter,
    DefaultLogicOptimizer,
    DefaultLoopOrderer,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
)

from .compiler import FinchJLCompiler

COMPILE_JULIA = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            DefaultLoopOrderer(DefaultLogicFormatter(LogicCompiler(FinchJLCompiler())))
        )
    )
)
