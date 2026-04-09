from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from ... import finch_assembly as asm
from ... import finch_notation as ntn
from ...algebra import StructFType, ffunc
from ...compile import looplets as lplt
from ...finch_assembly import parse_assembly
from ...interface.scalar import Scalar
from ..fiber_tensor import FiberTensorFields, FiberTensorFType, Level, LevelFType


class SparseListLevelFields(NamedTuple):
    lvl_asm: asm.AssemblyExpression  # assembly expression of the current level
    ptr_s: asm.Slot
    idx_s: asm.Slot
    next_lvl: NamedTuple


@dataclass(unsafe_hash=True)
class SparseListLevelFType(LevelFType, StructFType):
    _lvl_t: LevelFType
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
            ("ptr", self.buffer_factory(self.dimension_type)),
            ("idx", self.buffer_factory(self.dimension_type)),
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

    @property
    def ptr_type(self):
        return self.buffer_factory(self.dimension_type)

    @property
    def idx_type(self):
        return self.buffer_factory(self.dimension_type)

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

    @property
    def lvl_t(self):
        return self._lvl_t

    def from_numpy(self, shape, val):
        raise NotImplementedError("sparse list level doesn't support from_numpy")

    def level_asm_unpack(self, ctx, var_n, val) -> SparseListLevelFields:
        # Unpack ptr buffer
        ptr_buf = asm.Variable(f"{var_n}_ptr", self.ptr_type)
        ptr_buf_e = asm.GetAttr(val, asm.Literal("ptr"))
        ctx.exec(asm.Assign(ptr_buf, ptr_buf_e))
        ptr_s = asm.Slot(f"{var_n}_ptr_slot", self.ptr_type)
        ctx.exec(asm.Unpack(ptr_s, ptr_buf))

        # Unpack idx buffer
        idx_buf = asm.Variable(f"{var_n}_idx", self.idx_type)
        idx_buf_e = asm.GetAttr(val, asm.Literal("idx"))
        ctx.exec(asm.Assign(idx_buf, idx_buf_e))
        idx_s = asm.Slot(f"{var_n}_idx_slot", self.idx_type)
        ctx.exec(asm.Unpack(idx_s, idx_buf))

        return SparseListLevelFields(
            val,
            ptr_s,
            idx_s,
            self.lvl_t.level_asm_unpack(
                ctx, var_n, asm.GetAttr(val, asm.Literal("lvl"))
            ),
        )

    def level_asm_repack(self, ctx, lvl_fields):
        # TODO
        return super().level_asm_repack(ctx, lvl_fields)

    def level_lower_dim(self, ctx, lvl_fields: SparseListLevelFields, r):
        if r == 0:
            return asm.GetAttr(lvl_fields.lvl_asm, asm.Literal("dimension"))
        return self.lvl_t.level_lower_dim(ctx, lvl_fields.next_lvl, r - 1)

    def level_lower_declare(
        self, ctx, level_fields: SparseListLevelFields, init, op, shape, pos
    ):
        return self.lvl_t.level_lower_declare(
            ctx, level_fields.next_lvl, init, op, shape, pos
        )

    def level_lower_freeze(self, ctx, tns: SparseListLevelFields, op, pos):
        p_t = self.position_type
        lvl_ptr = tns.ptr_s
        lvl_idx = tns.idx_s
        pos_stop = asm.Variable("pos_stop", p_t)
        qos_stop = asm.Variable("qos_stop", p_t)
        p = asm.Variable("p", p_t)

        expr = """finch
        resize(lvl_ptr, pos_stop + 1)
        for (p in 0:pos_stop)
            lvl_ptr[p + 1] += lvl_ptr[p]
        end
        qos_stop = lvl_ptr[pos_stop] - 1
        resize(lvl_idx, qos_stop)
        """

        ctx.exec(parse_assembly(expr, locals()))
        return self.lvl_t.level_lower_freeze(ctx, tns.next_lvl, op, pos)

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
        resize_if_smaller = ...
        fill_range = ...
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
        tns: FiberTensorFields = stack.obj
        ft_ftype: FiberTensorFType = stack.type
        assert isinstance(tns.lvl_fields, SparseListLevelFields)
        ptr_s = tns.lvl_fields.ptr_s
        idx_s = tns.lvl_fields.idx_s
        next_lvl = tns.lvl_fields.next_lvl

        q = asm.Variable(ctx.freshen("q"), self.position_type)
        q_stop = asm.Variable(ctx.freshen("q_stop"), self.position_type)
        i_stop = asm.Variable(ctx.freshen("i_stop"), self.position_type)
        i_last = asm.Variable(ctx.freshen("i_last"), self.position_type)
        pos = tns.pos
        scalar = Scalar(self.fill_value, self.fill_value)

        tmp_locals = locals()

        def thunk_preamble(ctx, idx):
            expr = """finch
            q = ptr_s[pos]
            q_stop = ptr_s[pos + 1]
            if (q < q_stop)
                i_stop = idx_s[q]
                i_last = idx_s[q_stop - 1]
            else
                i_stop = 1
                i_last = 0
            end
            """
            return parse_assembly(expr, tmp_locals)

        def seek_fn(ctx, ext):
            code = """finch
            if (idx_s[q] < {start})
              q = scansearch(idx_s, {start}, q, q_stop - 1)
            end
            """.format(start=ctx.ctx(ext.get_start()))

            return parse_assembly(code, tmp_locals)

        def chunk_tail_fn(ctx, idx):
            pos_2 = asm.Variable(
                ctx.freshen(idx, f"_pos_{self.ndim - 1}"), self.position_type
            )
            ctx.exec(asm.Assign(pos_2, q))
            return lplt.Leaf(
                lambda ctx: ntn.Stack(
                    FiberTensorFields(
                        next_lvl, pos_2, tns.dirty_bit, tns.visited_idxs + (idx,)
                    ),
                    FiberTensorFType(ft_ftype.lvl_t.lvl_t),  # type: ignore[abstract]
                )
            )

        return lplt.Thunk(
            preamble=thunk_preamble,
            body=lambda ctx, ext: lplt.Sequence(
                head=lambda ctx, idx: lplt.Stepper(
                    preamble=lambda ctx: asm.Block(
                        (asm.Assign(i_stop, asm.Load(idx_s, q)),)
                    ),
                    stop=lambda ctx: ntn.Variable(i_stop.name, self.position_type),
                    chunk=lplt.Sequence(
                        head=lambda ctx, idx: lplt.Run(
                            lambda ctx, idx: lplt.Leaf(
                                lambda ctx: ntn.Stack(asm.Literal(scalar), scalar.ftype)
                            ),
                        ),
                        split=lambda ctx, ext: ext.get_end(),
                        tail=chunk_tail_fn,
                    ),
                    next=lambda ctx: asm.Block(
                        (
                            asm.Assign(
                                q,
                                asm.Call(asm.L(ffunc.add), (q, asm.L(self.p_t(1)))),
                            ),
                        )
                    ),
                    seek=seek_fn,
                ),
                split=lambda ctx, idx: ntn.Call(
                    ntn.L(ffunc.add),
                    (ntn.Variable(i_last.name, self.position_type), ntn.L(self.p_t(1))),
                ),
                tail=lambda ctx, idx: lplt.Run(
                    lambda ctx, idx: lplt.Leaf(
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
        return np.intp(0)

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
