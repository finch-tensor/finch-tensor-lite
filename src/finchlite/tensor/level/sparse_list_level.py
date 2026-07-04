from dataclasses import dataclass
from typing import Any

import numpy as np

from finchlite import finch_assembly as asm
from finchlite import finch_notation as ntn
from finchlite.algebra import FType, ImmutableStructFType, ffuncs, ftype, ftypes
from finchlite.compile import looplets as lplt
from finchlite.finch_assembly import parse_assembly
from finchlite.tensor.fiber_tensor import (
    FiberTensorFType,
    Level,
    LevelFType,
)
from finchlite.tensor.scalar import Scalar


@dataclass(unsafe_hash=True)
class SparseListLevelFType(LevelFType, ImmutableStructFType):
    _lvl_t: LevelFType
    dimension_type: FType = ftypes.intp

    def __post_init__(self):
        self.dimension_type = ftype(self.dimension_type)

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

    def construct(self, *, shape):
        """
        Creates an instance of SparseListLevel.

        Args:
            shape: The shape to be used for the level. (mandatory)
        Returns:
            An instance of DenseLevel.
        """
        lvl = self.lvl_t.construct(shape=shape[1:])
        return SparseListLevel(lvl, self.dimension_type(shape[0]))

    def __call__(self, val: Any) -> "SparseListLevel":
        """
        Convert a level to this sparse list level type.

        Args:
            val: A value to convert to this type.
        Returns:
            A SparseListLevel instance of this type.
        """
        raise NotImplementedError(
            f"Level conversion not yet implemented for {type(self).__name__}"
        )

    @property
    def lvl_t(self):
        return self._lvl_t

    def from_numpy(self, shape, val):
        raise NotImplementedError("sparse list level doesn't support from_numpy")

    def level_lower_dim(self, ctx, lvl, r):
        if r == 0:
            return asm.GetAttr(lvl, asm.Literal("dimension"))
        return self.lvl_t.level_lower_dim(
            ctx, asm.GetAttr(lvl, asm.Literal("lvl")), r - 1
        )

    def level_lower_declare(self, ctx, lvl, init, op, shape, pos):
        return self.lvl_t.level_lower_declare(
            ctx, asm.GetAttr(lvl, asm.Literal("lvl")), init, op, shape, pos
        )

    def level_lower_thaw(self, ctx, lvl, op, pos):
        return self.lvl_t.level_lower_thaw(
            ctx, asm.GetAttr(lvl, asm.Literal("lvl")), op, pos
        )

    def level_lower_freeze(self, ctx, lvl, op, pos):
        p_t = self.position_type
        lvl_ptr = asm.GetAttr(lvl, asm.Literal("ptr"))
        lvl_idx = asm.GetAttr(lvl, asm.Literal("idx"))
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
        return self.lvl_t.level_lower_freeze(
            ctx, asm.GetAttr(lvl, asm.Literal("lvl")), op, pos
        )

    def level_lower_increment(self, ctx, obj, op, val, pos):
        raise NotImplementedError(
            "SparseListLevelFType does not support level_lower_increment."
        )

    def level_lower_unwrap(self, ctx, obj, pos):
        raise NotImplementedError(
            "SparseListLevelFType does not support level_lower_unwrap."
        )

    def level_unfurl(
        self, ctx, stack: ntn.Fiber, ext, mode: ntn.AccessMode, proto, pos
    ):
        if not isinstance(stack.type, FiberTensorFType):
            raise TypeError(f"Expected FiberTensorFType, got: {stack.type}")
        tns = stack
        ft_ftype: FiberTensorFType = stack.type
        lvl_asm = ctx.fiber_level(tns)
        ptr_s = asm.GetAttr(lvl_asm, asm.Literal("ptr"))
        idx_s = asm.GetAttr(lvl_asm, asm.Literal("idx"))

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
            start = ctx.ctx(ext.get_start())

            code = f"""finch
            if (idx_s[q] < {start})
              q = scansearch(idx_s, {start}, q, q_stop - 1)
            end
            """

            return parse_assembly(code, tmp_locals | asm.get_vars_in_expr(start))

        def chunk_tail_fn(ctx, idx):
            pos_2 = asm.Variable(
                ctx.freshen(idx, f"_pos_{self.ndim - 1}"), self.position_type
            )
            ctx.exec(asm.Assign(pos_2, q))
            child_type = FiberTensorFType(ft_ftype.lvl_t.lvl_t)  # type: ignore[abstract]
            return lplt.Leaf(
                lambda ctx: ntn.Fiber(
                    tns.root,
                    ntn.Child(tns.lvl),
                    pos_2,
                    child_type,
                    (*tns.idxs, idx),
                )
            )

        return lplt.Thunk(
            preamble=thunk_preamble,
            body=lambda ctx, ext: lplt.Sequence(
                head=lambda ctx, idx: lplt.Stepper(
                    preamble=lambda ctx: asm.IfElse(
                        asm.Call(asm.L(ffuncs.lt), (q, q_stop)),
                        asm.Block((asm.Assign(i_stop, asm.Load(idx_s, q)),)),
                        asm.Block(
                            (
                                asm.Assign(
                                    i_stop,
                                    asm.GetAttr(lvl_asm, asm.Literal("dimension")),
                                ),
                            )
                        ),
                    ),
                    stop=lambda ctx: ntn.Variable(i_stop.name, self.position_type),
                    chunk=lplt.Sequence(
                        head=lambda ctx, idx: lplt.Run(scalar),
                        split=lambda ctx, ext: ntn.Variable(
                            i_stop.name, self.position_type
                        ),
                        tail=chunk_tail_fn,
                    ),
                    next=lambda ctx: asm.Block(
                        (
                            asm.Assign(
                                q,
                                asm.Call(asm.L(ffuncs.add), (q, asm.L(self.p_t(1)))),
                            ),
                        )
                    ),
                    seek=seek_fn,
                ),
                split=lambda ctx, idx: ntn.Call(
                    ntn.L(ffuncs.add),
                    (ntn.Variable(i_last.name, self.position_type), ext.get_unit()),
                ),
                tail=lambda ctx, idx: lplt.Run(scalar),
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
            self.ptr = self.lvl.buffer_type(len=0, dtype=self.lvl.position_type)
        if self.idx is None:
            self.idx = self.lvl.buffer_type(len=0, dtype=self.lvl.position_type)

    @property
    def stride(self) -> np.integer:
        return np.intp(0)

    @property
    def ftype(self) -> SparseListLevelFType:
        # mypy does not understand that dataclasses generate __hash__ and __eq__
        # https://github.com/python/mypy/issues/19799
        return SparseListLevelFType(self.lvl.ftype, ftype(self.dimension))  # type: ignore[abstract]

    @property
    def val(self) -> Any:
        return self.lvl.val

    def __str__(self):
        return f"SparseListLevel(lvl={self.lvl}, dim={self.dimension})"
