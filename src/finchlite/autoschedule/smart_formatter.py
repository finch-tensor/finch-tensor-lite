from __future__ import annotations

import logging
from abc import abstractmethod
from collections import OrderedDict
from typing import Any

import numpy as np

from finchlite import finch_logic as lgc
from finchlite.algebra import FType, TensorFType, ftype
from finchlite.finch_logic import LogicLoader, StatsFactory
from finchlite.finch_logic.tensor_stats import TensorStats
from finchlite.util.logging import LOG_LOGIC_POST_OPT

from .formatter import LogicFormatter
from .tensor_stats import StatsInterpreter

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


class SmartFormatter(LogicFormatter):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__(loader)

    @abstractmethod
    def get_tensor_ftype(
        self,
        fill_value: Any,
        shape_type: tuple[FType, ...],
        stats: TensorStats,
    ) -> TensorFType: ...

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        bindings = bindings.copy()
        stats_bindings: OrderedDict[lgc.Alias, TensorStats] = OrderedDict(stats)
        stats_interpreter = StatsInterpreter(stats_factory=stats_factory)
        shape_types = prgm.infer_shape_type(
            {var: val.shape_type for var, val in bindings.items()}
        )
        fill_values = prgm.infer_fill_value(
            {var: val.fill_value for var, val in bindings.items()}
        )

        def formatter(node: lgc.LogicStatement) -> lgc.LogicStatement:
            match node:
                case lgc.Plan(bodies):
                    return lgc.Plan(tuple(formatter(body) for body in bodies))
                case lgc.Query(lhs, rhs):
                    rhs_stats = stats_interpreter(rhs, stats_bindings)
                    if not isinstance(rhs_stats, TensorStats):
                        raise TypeError("Expected query RHS to produce TensorStats.")
                    stats_bindings[lhs] = rhs_stats

                    if lhs not in bindings:
                        shape_type = tuple(
                            ftype(dim) if dim is not None else ftype(np.intp)
                            for dim in shape_types[lhs]
                        )
                        bindings[lhs] = self.get_tensor_ftype(
                            fill_values[lhs],
                            shape_type,
                            rhs_stats,
                        )

                    match rhs:
                        case lgc.Reorder():
                            return node
                        case _:
                            return lgc.Query(lhs, lgc.Reorder(rhs, rhs.fields()))
                case lgc.Produces():
                    return node
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        prgm = formatter(prgm)

        logger.debug(prgm)

        return self.ctx(prgm, bindings, stats_bindings, stats_factory)

class FDFormatter(SmartFormatter):
    def get_tensor_ftype(
        self,
        fill_value: Any,
        shape_type: tuple[FType, ...],
        stats: TensorStats,
    ) -> TensorFType:
        assert isinstance(stats, FDStats)
        lvl = ElementLevel(fill_value)
        fields = stats.fields
        while fields:
            c = fields.poplast()
            dense = False
            for h in stats.dense_properties[c]:
                if h.subset(set(fields)):
                    dense = True
                    break
            if dense:
                lvl = DenseLevel(lvl)
            else:
                lvl = SparseHashLevel(lvl)
        return FiberTensor(lvl)