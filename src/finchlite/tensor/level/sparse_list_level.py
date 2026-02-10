import operator
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from finchlite.finch_assembly.parser import parse_assembly
from finchlite.interface.scalar import Scalar

from ... import finch_assembly as asm
from ... import finch_notation as ntn
from ...compile import LoopletContext
from ...compile import looplets as lplt
from ..fiber_tensor import FiberTensorFType, Level, LevelFType


class SparseListLevelSlots(NamedTuple):
    ptr_s: asm.Slot
    idx_s: asm.Slot


class SparseListLevelFields(NamedTuple):
    lvl: asm.Variable
    lvls_slots: tuple[SparseListLevelSlots, ...]
    pos: asm.Variable | asm.Literal
    op: asm.Literal
    dirty_bit: bool


@dataclass(unsafe_hash=True)
class SparseListLevelFType(LevelFType, asm.AssemblyStructFType):
    lvl_t: LevelFType
    dimension_type: Any = None

    @property
    def struct_name(self):
        return "SparseListLevelFType"

    @property
    def p_t(self):
        return self.position_type

    @property
    def struct_fields(self):
        return [
            ("lvl", self.lvl_t),
            ("dimension", self.dimension_type),
            ("stride", self.dimension_type),
            ("ptr", self.buffer_type),
            ("idx", self.buffer_type),
        ]

    def __post_init__(self):  # TODO: use different constructor instead
        if self.dimension_type is None:
            self.dimension_type = np.intp

    def __str__(self):
        return f"SparseListLevelFType({self.lvl_t})"

    @property
    def ndim(self):
        return 1 + self.lvl_t.ndim

    @property
    def fill_value(self):
        return self.lvl_t.fill_value

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the fibers.
        """
        return self.lvl_t.element_type

    @property
    def shape_type(self):
        """
        Returns the type of the shape of the fibers.
        """
        return (self.dimension_type, *self.lvl_t.shape_type)

    @property
    def position_type(self):
        """
        Returns the type of positions within the levels.
        """
        return self.lvl_t.position_type

    @property
    def buffer_type(self):
        return self.lvl_t.buffer_type

    @property
    def buffer_factory(self):
        """
        Returns the ftype of the buffer used for the fibers.
        """
        return self.lvl_t.buffer_factory

    def __call__(self, *, shape):
        """
        Creates an instance of SparseListLevel.

        Args:
            shape: The shape to be used for the level. (mandatory)
        Returns:
            An instance of DenseLevel.
        """
        lvl = self.lvl_t(shape=shape[1:])
        return SparseListLevel(lvl, self.dimension_type(shape[0]))

    def next_level(self):
        return self.lvl_t

    def from_numpy(self, shape, val):
        raise NotImplementedError("sparse list level doesn't support from_numpy")

    def get_fields_class(self, tns, lvls_slots, pos, op, dirty_bit):
        return SparseListLevelFields(tns, lvls_slots, pos, op, dirty_bit)

    def level_asm_unpack(self, ctx, var_n, val) -> tuple[asm.Slot, asm.Slot, asm.Slot]:
        # Unpack ptr buffer
        ptr_buf = asm.Variable(f"{var_n}_ptr", self.buffer_type)
        ptr_buf_e = asm.GetAttr(val, asm.Literal("ptr"))
        ctx.exec(asm.Assign(ptr_buf, ptr_buf_e))
        ptr_s = asm.Slot(f"{var_n}_ptr_slot", self.buffer_type)
        ctx.exec(asm.Unpack(ptr_s, ptr_buf))

        # Unpack idx buffer
        idx_buf = asm.Variable(f"{var_n}_idx", self.buffer_type)
        idx_buf_e = asm.GetAttr(val, asm.Literal("idx"))
        ctx.exec(asm.Assign(idx_buf, idx_buf_e))
        idx_s = asm.Slot(f"{var_n}_idx_slot", self.buffer_type)
        ctx.exec(asm.Unpack(idx_s, idx_buf))

        val_lvl = asm.GetAttr(val, asm.Literal("lvl"))
        return (SparseListLevelSlots(ptr_s, idx_s),) + self.lvl_t.level_asm_unpack(
            ctx, var_n, val_lvl
        )

    def level_lower_dim(self, ctx, obj, r):
        if r == 0:
            return asm.GetAttr(obj, asm.Literal("dimension"))
        obj = asm.GetAttr(obj, asm.Literal("lvl"))
        return self.lvl_t.level_lower_dim(ctx, obj, r - 1)

    def level_lower_declare(self, ctx, tns, init, op, shape, pos):
        return self.lvl_t.level_lower_declare(ctx, tns, init, op, shape, pos)

    def level_lower_freeze(self, ctx, tns: SparseListLevelFields, op, pos):
        p_t = self.position_type
        lvl_ptr = tns.lvls_slots[0].ptr_s
        lvl_idx = tns.lvls_slots[0].idx_s
        pos_stop = asm.Variable("pos_stop", p_t)
        qos_stop = asm.Variable("qos_stop", p_t)
        p = asm.Variable("p", p_t)

        expr = """finch
        resize(lvl_ptr, pos_stop + 1)
        for (p in 0:pos_stop)
            lvl_ptr[p + 1] += lvl_ptr[p]
        end
        qos_stop = lvl_ptr[pos_stop] - 1
        // some comment
        resize(lvl_idx, qos_stop)
        """

        ctx.exec(parse_assembly(expr, locals()))
        return self.lvl_t.level_lower_freeze(ctx, tns, op, pos)

    def level_lower_thaw(self, ctx, tns, op, pos):
        # TODO: implement
        return self.lvl_t.level_lower_thaw(ctx, tns, op, pos)

    def level_lower_increment(self, ctx, obj, val, pos):
        raise NotImplementedError(
            "SparseListLevelFType does not support level_lower_increment."
        )

    def level_lower_unwrap(self, ctx, obj, pos):
        raise NotImplementedError(
            "SparseListLevelFType does not support level_lower_unwrap."
        )

    def level_lower_assemble(self, ctx, tns, fill_value_type, pos_start, pos_stop):
        return asm.Block(
            (
                asm.Call(asm.Literal(resize_if_smaller), (tns.ptr_s, pos_stop)),
                asm.Call(
                    asm.Literal(fill_range),
                    (asm.Literal(fill_value_type(0)), pos_start, pos_stop),
                ),
            )
        )

    def level_unfurl(self, ctx, stack: asm.Stack, ext, mode, proto, pos):
        assert isinstance(stack.obj, SparseListLevelFields)
        tns: SparseListLevelFields = stack.obj
        assert isinstance(stack.type, FiberTensorFType)
        ft_ftype: FiberTensorFType = stack.type
        assert isinstance(tns.lvls_slots[0], SparseListLevelSlots)
        lvl_ptr = tns.lvls_slots[0].ptr_s
        lvl_idx = tns.lvls_slots[0].idx_s

        q = asm.Variable(ctx.freshen("q"), self.position_type)
        q_stop = asm.Variable(ctx.freshen("q_stop"), self.position_type)
        i = asm.Variable(ctx.freshen("i"), self.position_type)
        i1 = asm.Variable(ctx.freshen("i1"), self.position_type)
        pos = tns.pos
        scalar = Scalar(self.fill_value, self.fill_value)

        tmp_locals = locals()

        def thunk_preamble(ctx: LoopletContext, idx):
            expr = """finch
            q = lvl_ptr[pos]
            q_stop = lvl_ptr[pos + 1]
            if (q < q_stop)
                i = lvl_idx[q]
                i1 = lvl_idx[q_stop - 1]
            else
                i = 1
                i1 = 0
            end
            """
            return parse_assembly(expr, tmp_locals)

        def seek_fn(ctx, ext):
            code = f"""finch
            if (lvl_idx[q] < {0})
              q = scansearch(lvl_idx, {0}, q, q_stop - 1)
            end
            """  # ormat(start=ext.result_format.get_start(ext))

            return parse_assembly(code, tmp_locals)

        def chunk_tail_fn(ctx, idx):
            pos_2 = asm.Variable(
                ctx.freshen(idx, f"_pos_{self.ndim - 1}"), self.position_type
            )
            return ntn.Stack(
                self.lvl_t.get_fields_class(
                    asm.GetAttr(tns.lvl, asm.Literal("lvl")),
                    tns.lvls_slots[1:],
                    pos_2,
                    tns.op,
                    tns.dirty_bit,
                ),
                FiberTensorFType(ft_ftype.lvl_t.next_level()),  # type: ignore[abstract]
            )

        return lplt.Thunk(
            preamble=thunk_preamble,
            body=lambda ctx, ext: lplt.Sequence(
                head=lambda ctx, idx: lplt.Stepper(
                    preamble=lambda ctx, ext: asm.Block(
                        (asm.Assign(i, asm.Load(tns.idx_s, q))),
                    ),
                    stop=lambda ctx, ext: i,
                    chunk=lplt.Spike(
                        body=lambda ctx, idx: lplt.Run(
                            lambda ctx: lplt.Leaf(
                                lambda ctx: ntn.Stack(asm.Literal(scalar), scalar.ftype)
                            ),
                        ),
                        tail=chunk_tail_fn,
                    ),
                    next=lambda ctx, ext: asm.Block(
                        (
                            asm.Assign(
                                q,
                                asm.Call(
                                    asm.Literal(operator.add),
                                    (q, asm.Literal(self.position_type(1))),
                                ),
                            ),
                        )
                    ),
                    seek=seek_fn,
                ),
                split=lambda ctx, idx, visited_idx: i1,
                tail=lambda ctx, idx: lplt.Run(
                    lambda ctx: lplt.Leaf(
                        lambda ctx: ntn.Stack(asm.Literal(scalar), scalar.ftype)
                    )
                ),
            ),
        )

    def from_fields(self, lvl, dimension, ptr, idx) -> "SparseListLevel":
        return SparseListLevel(lvl, dimension, ptr, idx)


def sparse_list(lvl_t, dimension_type=None):
    return SparseListLevelFType(lvl_t, dimension_type)


@dataclass
class SparseListLevel(Level):
    """
    A class representing sparse list level.
    """

    lvl: Level
    dimension: np.integer
    ptr: Any | None = None
    idx: Any | None = None

    @property
    def shape(self) -> tuple:
        return (self.dimension, *self.lvl.shape)

    def __post_init__(self):
        if self.ptr is None:
            self.ptr = self.lvl.buffer_type(len=0, dtype=self.lvl.position_type())
        if self.idx is None:
            self.idx = self.lvl.buffer_type(len=0, dtype=self.lvl.position_type())

    @property
    def stride(self) -> np.integer:
        stride = self.lvl.stride
        if self.lvl.ndim == 0:
            return stride
        return self.lvl.shape[0] * stride

    @property
    def ftype(self) -> SparseListLevelFType:
        # mypy does not understand that dataclasses generate __hash__ and __eq__
        # https://github.com/python/mypy/issues/19799
        return SparseListLevelFType(self.lvl.ftype, type(self.dimension))  # type: ignore[abstract]

    @property
    def val(self) -> Any:
        return self.lvl.val

    def __str__(self):
        return f"SparseListLevel(lvl={self.lvl}, dim={self.dimension})"
