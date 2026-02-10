from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from ... import finch_assembly as asm
from ...codegen import NumpyBufferFType
from ...symbolic import FType, ftype
from ..fiber_tensor import Level, LevelFType


class ElementLevelSlots(NamedTuple):
    buf_s: asm.Slot


class ElementLevelFields(NamedTuple):
    lvl: asm.Variable
    lvls_slots: tuple[ElementLevelSlots]
    pos: asm.Variable | asm.Literal
    op: asm.Literal
    dirty_bit: bool


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

    def __call__(self, shape, *, val=None):
        """
        Creates an instance of ElementLevel with the given ftype.

        Args:
            shape: Should be always `()`, used for validation.
            val: The value to store in the ElementLevel instance.
        Returns:
            An instance of ElementLevel.
        """
        # Wrap numpy arrays in NumpyBuffer and flatten, similar to BufferizedNDArray
        if val is not None and isinstance(val, np.ndarray):
            from ...codegen import NumpyBuffer

            val = NumpyBuffer(np.asarray(val).reshape(-1))
        if len(shape) != 0:
            raise ValueError("ElementLevelFType must be called with an empty shape.")
        return ElementLevel(self, val)

    def __str__(self):
        return f"ElementLevelFType(fv={self.fill_value})"

    @property
    def ndim(self):
        return 0

    @property
    def shape_type(self):
        return ()

    def from_fields(self, val=None) -> "ElementLevel":
        # Wrap numpy arrays in NumpyBuffer and flatten, similar to BufferizedNDArray
        if val is not None and isinstance(val, np.ndarray):
            from ...codegen import NumpyBuffer

            val = NumpyBuffer(np.asarray(val).reshape(-1, copy=False))
        return ElementLevel(_format=self, _val=val)

    def level_asm_unpack(self, ctx, var_n, val) -> tuple[ElementLevelSlots]:
        buf = asm.Variable(f"{var_n}_buf", self.buffer_type)
        buf_e = asm.GetAttr(val, asm.Literal("val"))
        ctx.exec(asm.Assign(buf, buf_e))
        buf_s = asm.Slot(f"{var_n}_buf_slot", self.buffer_type)
        ctx.exec(asm.Unpack(buf_s, buf))

        return (ElementLevelSlots(buf_s),)

    def level_asm_repack(self, ctx, lvls_slots):
        slots = lvls_slots[0]
        ctx.exec(asm.Repack(slots.buf_s))

    def get_fields_class(self, tns, lvls_slots, pos, op, dirty_bit):
        return ElementLevelFields(tns, lvls_slots, pos, op, dirty_bit)

    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        i_var = asm.Variable("i", self.buffer_type.length_type)
        body = asm.Store(tns, i_var, asm.Literal(init.val))
        ctx.exec(asm.ForLoop(i_var, asm.Literal(np.intp(0)), asm.Length(tns), body))

    def level_lower_unwrap(self, ctx, obj, pos):
        assert isinstance(obj, ElementLevelFields)
        return asm.Load(obj.lvls_slots[0].buf_s, pos)

    def level_lower_increment(self, ctx, obj, val, pos):
        assert isinstance(obj, ElementLevelFields)
        buf_s = obj.lvls_slots[0].buf_s
        lowered_pos = asm.Variable(pos.name, pos.type)
        ctx.exec(
            asm.Store(
                buf_s,
                lowered_pos,
                asm.Call(
                    asm.Literal(obj.op.val),
                    [asm.Load(buf_s, lowered_pos), val],
                ),
            )
        )

    def level_lower_freeze(self, ctx, tns, op):
        return tns

    def level_lower_thaw(self, ctx, tns, op):
        return tns

    def level_lower_dim(self, ctx, obj, r):
        raise NotImplementedError("ElementLevelFType does not support level_lower_dim.")

    def level_unfurl(self, ctx, tns, ext, mode, proto, pos):
        raise NotImplementedError("ElementLevelFType does not support level_unfurl.")

    def next_level(self):
        raise NotImplementedError("ElementLevelFType does not support next_level.")

    def from_numpy(self, shape, val):
        return self(shape=shape, val=val)


def element(
    fill_value=None,
    element_type=None,
    position_type=None,
    buffer_factory=NumpyBufferFType,
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

    _format: ElementLevelFType = field(repr=False)
    _val: Any | None = None

    def __post_init__(self):
        if self._val is None:
            self._val = self._format.buffer_type(
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
    def val(self) -> Any:
        return self._val

    def __str__(self):
        return f"ElementLevel(val={self._val})"
