from finchlite.algebra.tensor import TensorFType
import numpy as np

from finchlite.finch_assembly import AssemblyLibrary
from finchlite.finch_logic.nodes import TableValueFType

from .. import finch_logic as lgc
from ..codegen import NumpyBufferFType
from ..compile import BufferizedNDArrayFType
from ..finch_assembly import TupleFType
from ..finch_logic import LogicLoader, MockLogicLoader
from ..symbolic import gensym


class LogicFormatter(LogicLoader):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.loader = loader

    def __call__(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
    ) -> tuple[
        AssemblyLibrary, lgc.LogicStatement, dict[lgc.Alias, TensorFType]
    ]:
        bindings = bindings.copy()
        shape_types = prgm.infer_shape_type(
            {var: val.shape_type for var, val in bindings.items()}
        )
        element_types = prgm.infer_element_type(
            {var: val.element_type for var, val in bindings.items()}
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
                            ndim=np.intp(len(shape_type)),
                            dimension_type=TupleFType(
                                struct_name=gensym("ugh"), struct_formats=shape_type
                            ),
                        )
                        # tns = NDArrayFType(element_type, np.intp(len(shape_type)))
                        bindings[lhs] = tns
                case lgc.Produces(_):
                    pass
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        formatter(prgm)

        lib, prgm, bindings = self.loader(prgm, bindings)
        return lib, prgm, bindings
