import logging
from abc import abstractmethod
from typing import Any

import numpy as np

from .. import finch_logic as lgc
from ..algebra import TensorFType
from ..codegen import NumpyBufferFType
from ..compile import BufferizedNDArrayFType
from ..finch_assembly import AssemblyLibrary, TupleFType
from ..finch_logic import LogicLoader, MockLogicLoader, StatsFactory, TensorStats
from ..symbolic import gensym
from ..util.logging import LOG_LOGIC_POST_OPT

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


class LogicFormatter(LogicLoader):
    def __init__(
        self,
        loader: LogicLoader | None = None,
    ):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.loader = loader

    @abstractmethod
    def get_output_tns_ftype(self, fill_value: Any, shape_type: tuple[Any, ...]):
        """
        Return the FType of the output tensor produced within the
        autoscheduler.
        """
        ...

    def __call__(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, "TensorStats"],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        dict[lgc.Alias, tuple[lgc.Field | None, ...]],
    ]:
        bindings = bindings.copy()
        shape_types = prgm.infer_shape_type(
            {var: val.shape_type for var, val in bindings.items()}
        )
        fill_values = prgm.infer_fill_value(
            {var: val.fill_value for var, val in bindings.items()}
        )

        def formatter(node: lgc.LogicStatement):
            match node:
                case lgc.Plan(bodies):
                    for body in bodies:
                        formatter(body)
                case lgc.Query(lhs, _):
                    if lhs not in bindings:
                        shape_type = tuple(
                            dim if dim is not None else np.intp
                            for dim in shape_types[lhs]
                        )

                        tns = self.get_output_tns_ftype(fill_values[lhs], shape_type)

                        bindings[lhs] = tns
                case lgc.Produces(_):
                    pass
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        formatter(prgm)

        logger.debug(prgm)

        lib, bindings, shape_vars = self.loader(
            prgm,
            bindings,
            stats=stats,
            stats_factory=stats_factory,
        )
        return lib, bindings, shape_vars


class DefaultLogicFormatter(LogicFormatter):
    def __init__(
        self,
        loader: LogicLoader | None = None,
    ):
        super().__init__(loader)

    def get_output_tns_ftype(self, fill_value: Any, shape_type: tuple[Any, ...]):
        return BufferizedNDArrayFType(
            buffer_type=NumpyBufferFType(type(fill_value)),
            ndim=len(shape_type),
            dimension_type=TupleFType(
                struct_name=gensym("tuple", sep="_"),
                struct_formats=shape_type,
            ),
        )
