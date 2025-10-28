from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import numpy as np

from ... import finch_assembly as asm
from ...codegen import NumpyBufferFType
from ...symbolic import FType, ftype
from ..fiber_tensor import Level, LevelFType


class ElementLevelFields(NamedTuple):
    lvl: asm.Variable
    buf_s: NumpyBufferFType
    nind: int
    pos: asm.AssemblyNode
    op: Any


@dataclass(unsafe_hash=True)
class ElementLevelFType(LevelFType, asm.AssemblyStructFType):
    fill_value: Any = None
    element_type: type | FType | None = None
    position_type: type | FType | None = None
    buffer_factory: Any = NumpyBufferFType
    buffer_type: Any = None

    @property
    def struct_name(self):
        return "ElementLevelFType"

    @property
    def struct_fields(self):
        return [
            ("val", self.buffer_type),
        ]

    def __post_init__(self):
        if self.element_type is None:
            self.element_type = ftype(self.fill_value)
        if self.buffer_type is None:
            self.buffer_type = self.buffer_factory(self.element_type)
        if self.position_type is None:
            self.position_type = np.intp
        self.element_type = self.buffer_type.element_type
        self.fill_value = self.element_type(self.fill_value)

    def __call__(self, shape=(), val=None):
        """
        Creates an instance of ElementLevel with the given ftype.
        Args:
            shape: Should be always `()`, used for validation.
            val: The value to store in the ElementLevel instance.
        Returns:
            An instance of ElementLevel.
        """
        if len(shape) != 0:
            raise ValueError("ElementLevelFType must be called with an empty shape.")
        return ElementLevel(self, val)

    def __str__(self):
        return f"ElementLevelFType(fv={self.fill_value})"

    @property
    def ndim(self):
        return 0

    def from_kwargs(self, **kwargs) -> "ElementLevelFType":
        f_v = kwargs.get("fill_value", self.fill_value)
        e_t = kwargs.get("element_type", self.element_type)
        p_t = kwargs.get("position_type", self.position_type)
        b_f = kwargs.get("buffer_factory", self.buffer_factory)
        v_f = kwargs.get("buffer_type", self.buffer_type)
        return ElementLevelFType(f_v, e_t, p_t, b_f, v_f)

    def to_kwargs(self):
        return asdict(self)

    @property
    def shape_type(self):
        return ()

    def asm_unpack(self, ctx, var_n, val):
        buf = asm.Variable(f"{var_n}_buf", self.buffer_type)
        buf_e = asm.GetAttr(val, asm.Literal("val"))
        ctx.exec(asm.Assign(buf, buf_e))
        buf_s = asm.Slot(f"{var_n}_buf_slot", self.buffer_type)
        ctx.exec(asm.Unpack(buf_s, buf))
        return buf_s

    def get_fields_class(self, tns, buf_s, nind, pos, op):
        return ElementLevelFields(tns, buf_s, nind, pos, op)

    def lower_declare(self, ctx, tns, init, op, shape):
        i_var = asm.Variable("i", self.buffer_type.length_type)
        body = asm.Store(tns, i_var, asm.Literal(init.val))
        ctx.exec(asm.ForLoop(i_var, asm.Literal(np.intp(0)), asm.Length(tns), body))

    def lower_unwrap(self, ctx, obj):
        return asm.Load(obj.buf_s, obj.pos)

    def lower_increment(self, ctx, obj, val):
        lowered_pos = asm.Variable(obj.pos.name, obj.pos.type)
        ctx.exec(
            asm.Store(
                obj.buf_s,
                lowered_pos,
                asm.Call(
                    asm.Literal(obj.op.val),
                    [asm.Load(obj.buf_s, lowered_pos), val],
                ),
            )
        )

    def lower_freeze(self, ctx, tns, op):
        raise NotImplementedError("ElementLevelFType does not support lower_freeze.")

    def lower_thaw(self, ctx, tns, op):
        raise NotImplementedError("ElementLevelFType does not support lower_thaw.")

    def unfurl(self, ctx, tns, ext, mode, proto):
        raise NotImplementedError("ElementLevelFType does not support unfurl.")


def element(
    fill_value=None,
    element_type=None,
    position_type=None,
    buffer_factory=None,
    buffer_type=None,
):
    """
    Creates an ElementLevelFType with the given parameters.

    Args:
        fill_value: The value to be used as the fill value for the level.
        element_type: The type of elements stored in the level.
        position_type: The type of positions within the level.
        buffer_factory: The factory used to create buffers for the level.
        buffer_type: Format of the value stored in the level.

    Returns:
        An instance of ElementLevelFType.
    """
    return ElementLevelFType(
        fill_value=fill_value,
        element_type=element_type,
        position_type=position_type,
        buffer_factory=buffer_factory,
        buffer_type=buffer_type,
    )


@dataclass
class ElementLevel(Level):
    """
    A class representing the leaf level of Finch tensors.
    """

    _format: ElementLevelFType
    val: Any | None = None

    def __post_init__(self):
        if self.val is None:
            self.val = self._format.buffer_type(
                len=0, dtype=self._format.element_type()
            )

    @property
    def shape(self) -> tuple:
        return ()

    @property
    def stride(self) -> np.integer:
        return np.intp(1)  # TODO: add dimension_type to element_level.py

    @property
    def ftype(self) -> ElementLevelFType:
        return self._format

    @property
    def buf(self):
        return self.val
