from typing import Any

from finchlite.autoschedule import (
    DefaultLogicOptimizer,
    LogicCompiler,
    LogicExecutor,
    LogicFormatter,
    LogicNormalizer,
    LogicStandardizer,
)
from finchlite.finch_logic import LogicLoader

from .compiler import FinchJLCompiler
from .levels import DenseFormat, ElementFormat
from .tensor import FinchJLTensorFType


class FinchJLLogicFormatter(LogicFormatter):
    def __init__(
        self,
        loader: LogicLoader | None = None,
    ):
        super().__init__(loader)

    def get_output_tns_ftype(self, fill_value: Any, shape_type: tuple[Any, ...]):
        lvl = ElementFormat(fill_value)
        for _ in shape_type:
            lvl = DenseFormat(lvl)
        return FinchJLTensorFType(lvl)


COMPILE_JULIA = LogicNormalizer(
    LogicExecutor(
        DefaultLogicOptimizer(
            LogicStandardizer(FinchJLLogicFormatter(LogicCompiler(FinchJLCompiler())))
        )
    )
)
