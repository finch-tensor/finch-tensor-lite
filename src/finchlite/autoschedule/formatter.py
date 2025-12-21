from typing import Any

import numpy as np

from .. import finch_logic as lgc
from ..codegen import NumpyBufferFType
from ..compile import BufferizedNDArrayFType
from ..finch_assembly import AssemblyLibrary, TupleFType
from ..finch_logic import LogicLoader, MockLogicLoader, TableValueFType
from ..symbolic import gensym


class LogicFormatter(LogicLoader):
    """
    Docstring for LogicFormatter
    """

    def __init__(self, loader: LogicLoader | None = None):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.loader = loader

    def __call__(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, lgc.TableValueFType],
        *,
        debug_ctx: dict[str, Any] | None = None,
    ) -> tuple[
        AssemblyLibrary, lgc.LogicStatement, dict[lgc.Alias, lgc.TableValueFType]
    ]:
        bindings = bindings.copy()
        fields = prgm.infer_fields({var: val.idxs for var, val in bindings.items()})
        shape_types = prgm.infer_shape_type(
            {var: val.tns.shape_type for var, val in bindings.items()}, fields
        )
        element_types = prgm.infer_element_type(
            {var: val.tns.element_type for var, val in bindings.items()}
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

                        # TODO: This constructor is awful
                        # TODO: bufferized ndarray seems broken
                        tns = BufferizedNDArrayFType(
                            buffer_type=NumpyBufferFType(element_types[lhs]),
                            ndim=np.intp(len(fields[lhs])),
                            dimension_type=TupleFType(
                                struct_name=gensym("ugh"), struct_formats=shape_type
                            ),
                        )
                        # tns = NDArrayFType(element_type, np.intp(len(shape_type)))
                        bindings[lhs] = TableValueFType(tns, fields[lhs])
                case lgc.Produces(_):
                    pass
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        formatter(prgm)

        lib, prgm, bindings = self.loader(prgm, bindings)
        return lib, prgm, bindings
