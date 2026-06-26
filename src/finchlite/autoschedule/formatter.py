import logging
from abc import abstractmethod
from typing import Any

import numpy as np

from finchlite import finch_logic as lgc
from finchlite.algebra import FType, TensorFType, TupleFType, ftype
from finchlite.autoschedule.stages import LoopOrderedForm
from finchlite.codegen import NumpyBufferFType
from finchlite.finch_logic import (
    LogicLoader,
    MockLogicLoader,
    StatsFactory,
    TensorStats,
)
from finchlite.tensor import BufferizedNDArrayFType
from finchlite.util.logging import LOG_LOGIC_POST_OPT

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


class LogicFormatter(LogicLoader):
    pass


class DefaultLogicFormatter(LogicFormatter):
class LogicFormatter(LoopOrderedForm, LogicLoader):
    def __init__(
        self,
        loader: LogicLoader | None = None,
    ):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.ctx = loader

    @abstractmethod
    def get_output_tns_ftype(self, fill_value: Any, shape_type: tuple[FType, ...]):
        """
        Return the FType of the output tensor produced within the
        autoscheduler.
        """
        ...

    def lower(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, "TensorStats"],
        stats_factory: StatsFactory,
    ):
        bindings = bindings.copy()
        shape_types = prgm.infer_shape_type(
            {var: val.shape_type for var, val in bindings.items()}
        )
        fill_values = prgm.infer_fill_value(
            {var: val.fill_value for var, val in bindings.items()}
        )

        def formatter(node: lgc.LogicStatement) -> lgc.LogicStatement:
            match node:
                case lgc.Plan(bodies):
                    new_bodies = tuple(formatter(body) for body in bodies)
                    return lgc.Plan(new_bodies)
                case lgc.Query(lhs, rhs):
                    if lhs not in bindings:
                        shape_type = tuple(
                            ftype(dim) if dim is not None else ftype(np.intp)
                            for dim in shape_types[lhs]
                        )

                        tns = self.get_output_tns_ftype(fill_values[lhs], shape_type)

                        bindings[lhs] = tns
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

        return self.ctx(prgm, bindings, stats, stats_factory)


class BufferizedNDArrayFormatter(DefaultLogicFormatter):
    def __init__(
        self,
        loader: LogicLoader | None = None,
    ):
        super().__init__(loader)

    def get_output_tns_ftype(self, fill_value: Any, shape_type: tuple[FType, ...]):
        fill_ftype = ftype(
            fill_value.dtype if isinstance(fill_value, np.ndarray) else fill_value
        )
        return BufferizedNDArrayFType(
            buffer_type=NumpyBufferFType(fill_ftype),
            ndim=len(shape_type),
            dimension_type=TupleFType.from_tuple(shape_type),
        )
