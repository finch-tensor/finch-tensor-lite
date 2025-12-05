import numpy as np

from finchlite.algebra.tensor import NDArrayFType
from finchlite.finch_assembly import AssemblyLibrary
from finchlite.finch_logic.nodes import TableValueFType

from .. import finch_logic as lgc
from ..finch_logic import LogicLoader
from .fakecompiler import FakeLogicCompiler


class LogicFormatterContext:
    def __init__(self, bindings: dict[lgc.Alias, lgc.TableValueFType] | None = None):
        if bindings is None:
            bindings = {}
        self.bindings: dict[lgc.Alias, lgc.TableValueFType] = bindings.copy()
        self.fields = {idx: val.idxs for idx, val in bindings.items()}
        self.shape_types = {idx: val.tns.shape_type for idx, val in bindings.items()}
        self.element_types = {
            idx: val.tns.element_type for idx, val in bindings.items()
        }
        self.fill_values = {idx: val.tns.fill_value for idx, val in bindings.items()}

    def __call__(
        self, node: lgc.LogicStatement
    ) -> dict[lgc.Alias, lgc.TableValueFType]:
        match node:
            case lgc.Plan(bodies):
                for body in bodies:
                    self(body)
            case lgc.Query(lhs, rhs):
                fields = rhs.fields(self.fields)
                element_type = rhs.element_type(self.element_types)
                shape_type = rhs.shape_type(self.shape_types, self.fields)
                fill_value = rhs.fill_value(self.fill_values)
                if lhs in self.bindings:
                    if self.bindings[lhs].tns.shape_type != shape_type:
                        raise ValueError(
                            f"Shape type mismatch for {lhs}: "
                            f"{self.bindings[lhs].tns.shape_type} vs {shape_type}"
                        )
                    if self.bindings[lhs].tns.element_type != element_type:
                        raise ValueError(
                            f"Element type mismatch for {lhs}: "
                            f"{self.bindings[lhs].tns.element_type} vs {element_type}"
                        )
                    if self.bindings[lhs].tns.fill_value != fill_value:
                        raise ValueError(
                            f"Fill value mismatch for {lhs}: "
                            f"{self.bindings[lhs].tns.fill_value} vs {fill_value}"
                        )
                    if fields != self.bindings[lhs].idxs:
                        raise ValueError(
                            f"Field mismatch for {lhs}: "
                            f"{self.bindings[lhs].idxs} vs {fields}"
                        )
                else:
                    self.fields[lhs] = fields
                    self.shape_types[lhs] = shape_type
                    self.element_types[lhs] = element_type
                    self.fill_values[lhs] = fill_value

                    shape_type = tuple(
                        dim if dim is not None else np.intp for dim in shape_type
                    )

                    # TODO: This constructor is awful
                    # TODO: bufferized ndarray seems broken
                    # tns = BufferizedNDArrayFType(
                    #     buffer_type=NumpyBufferFType(element_type),
                    #     ndim=np.intp(len(fields)),
                    #     dimension_type=TupleFType(
                    #         struct_name=gensym("ugh"), struct_formats=shape_type
                    #     ),
                    # )
                    tns = NDArrayFType(element_type, np.intp(len(shape_type)))
                    self.bindings[lhs] = TableValueFType(tns, fields)
            case lgc.Produces(_):
                pass
            case _:
                raise ValueError(f"Unsupported logic statement for formatting: {node}")
        return self.bindings


class LogicFormatter(LogicLoader):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__()
        if loader is None:
            loader = FakeLogicCompiler()
        self.loader = loader

    def __call__(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, lgc.TableValueFType] | None = None,
    ) -> tuple[AssemblyLibrary, dict[lgc.Alias, lgc.TableValueFType]]:
        bindings = LogicFormatterContext(bindings)(prgm)
        lib, bindings = self.loader(prgm, bindings)
        return lib, bindings
