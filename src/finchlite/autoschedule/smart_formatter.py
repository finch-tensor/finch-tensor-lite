from __future__ import annotations

import logging
from abc import abstractmethod
from collections import OrderedDict
from typing import Any

from finchlite import finch_logic as lgc
from finchlite.algebra import FType, TensorFType, ftype, ftypes
from finchlite.finch_logic import LogicLoader, StatsFactory
from finchlite.finch_logic.tensor_stats import TensorStats
from finchlite.tensor import dense, element, fiber_tensor, sparse_hash
from finchlite.util.logging import LOG_LOGIC_POST_OPT

from .formatter import LogicFormatter
from .tensor_stats import FDStats, StatsInterpreter

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
                            ftype(dim) if dim is not None else ftypes.intp
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
        if not isinstance(stats, FDStats):
            raise TypeError("FDFormatter requires FDStats.")
        if len(shape_type) != len(stats.index_order):
            raise ValueError(
                f"Got {len(shape_type)} shape dimensions for "
                f"{len(stats.index_order)} stats dimensions."
            )

        fill_ftype = ftype(fill_value)
        lvl = element(fill_value, fill_ftype)
        for dim in reversed(range(len(stats.index_order))):
            field = stats.index_order[dim]
            outer_fields = frozenset(stats.index_order[:dim])
            required_fields = outer_fields | {field}
            is_dense = any(
                required_fields.issubset(dense_fields)
                for dense_fields in stats.dense_props
            )
            if is_dense:
                lvl = dense(lvl, shape_type[dim])
            else:
                lvl = sparse_hash(lvl, shape_type[dim], single_writer=False)

        return fiber_tensor(lvl)
