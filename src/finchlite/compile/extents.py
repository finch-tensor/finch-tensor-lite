from dataclasses import dataclass
from enum import Enum
from operator import add, sub
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import register_property
from ..finch_assembly import AssemblyStructFType
from ..symbolic import PostOrderDFS
from ..util import qual_str
from .lower import FinchCompileError, HaltState, lower_looplets


@dataclass(eq=True, frozen=True)
class Extent:
    """
    A class to represent the extent of a loop variable.
    This is used to define the start and end values of a loop.
    """

    start: Any
    end: Any

    def loop(self, ctx, idx, body):
        for idx_e in range(self.start, self.end):
            # Create a new scope for each iteration
            ctx_2 = ctx.scope(loop_state=HaltState())
            # Assign the loop variable
            ctx_2.bindings[idx.name] = idx.type_(idx_e)
            # Execute the body of the loop
            ctx_2(body)

    @property
    def ftype(self):
        return ExtentFType(
            np.asarray(self.start).dtype.type, np.asarray(self.end).dtype.type
        )


register_property(
    Extent,
    "__call__",
    "return_type",
    lambda op, x, y: ExtentFType(x, y),  # type: ignore[abstract]
)


@dataclass(eq=True, frozen=True)
class ExtentFields:
    start: Any
    end: Any


@dataclass(eq=True, frozen=True)
class ExtentFType(AssemblyStructFType):
    start: Any
    end: Any

    def __repr__(self):
        return f"ExtentFType(start={qual_str(self.start)}, end={qual_str(self.end)})"

    @classmethod
    def stack(cls, start, end):
        return ntn.Stack(
            ExtentFields(start, end),
            ExtentFType(start.result_format, end.result_format),
        )

    @property
    def struct_name(self):
        return "Extent"

    @property
    def struct_fields(self):
        return [("start", np.intp), ("end", np.intp)]

    def from_fields(self, start, stop) -> "Extent":
        return Extent(start, stop)

    def __call__(self, *args):
        raise TypeError(f"{self.struct_name} is not callable")

    def get_start(self, ext):
        match ext:
            case asm.Call(asm.Literal(op), (start, _)) if op is Extent:
                return start
            case _:
                return asm.GetAttr(ext, asm.Literal("start"))

    def get_end(self, ext):
        match ext:
            case asm.Call(asm.Literal(op), (_, end)) if op is Extent:
                return end
            case _:
                return asm.GetAttr(ext, asm.Literal("end"))

    def get_unit(self, ext):
        # TODO: fix
        return ntn.Literal(np.intp(1))

    def get_measure(self, ext):
        return ntn.Call(ntn.Literal(sub), (ext.get_end(), ext.get_start()))

    def lower_loop(self, ctx, idx, visited_idxs, ext, body):
        """
        Lower a loop with the given index and body.
        This is used to compile the loop into assembly.
        """
        lower_looplets(ctx, idx, visited_idxs, ext, body)
        return

    def bound_below(self, size) -> "ExtentFType":
        return self._bound_ext(size, max)

    def bound_above(self, size) -> "ExtentFType":
        return self._bound_ext(size, min)

    def _bound_ext(self, size, func) -> "ExtentFType":
        return ExtentFType(
            self.start,
            ntn.Cached(
                self.end,
                ntn.Call(
                    ntn.Literal(func),
                    (self.end, ntn.Call(ntn.Literal(add), (self.start, size))),
                ),
            ),
        )

    def default_loop(self, ctx, idx, ext, body):
        def assert_lowered(node):
            match node:
                case ntn.Access(_, _, (j, *_)):
                    if j == idx:
                        raise FinchCompileError(
                            node, f"Access with {j} should have been lowered already"
                        )
            return

        map(assert_lowered, PostOrderDFS(body))

        idx = asm.Variable(ctx.freshen(idx.name), idx.result_format)
        ctx_2 = ctx.scope()
        ctx_2.bindings[idx.name] = idx
        ctx_2(body)
        body_3 = asm.Block(ctx_2.emit())
        ctx.exec(
            asm.ForLoop(
                idx,
                self.get_start(ext),
                self.get_end(ext),
                body_3,
            )
        )
        return


@dataclass(eq=True, frozen=True)
class SingletonExtent:
    idx: Any

    def loop(self, ctx, idx, body):
        # Create a new scope for each iteration
        ctx_2 = ctx.scope(loop_state=HaltState())
        # Assign the loop variable
        ctx_2.bindings[idx.name] = idx.type_(self.idx)
        # Execute the body of the loop
        ctx_2(body)


@dataclass(eq=True, frozen=True)
class SingletonExtentFields:
    idx: Any


@dataclass(eq=True, frozen=True)
class SingletonExtentFType:
    idx: Any

    @classmethod
    def stack(cls, idx):
        return ntn.Stack(
            SingletonExtentFields(idx),
            SingletonExtentFType(idx.result_format),
        )

    def get_start(self, ext):
        return asm.GetAttr(ext, "idx")

    def get_end(self, ext):
        return asm.GetAttr(ext, "idx")

    def lower_loop(self, ctx, idx, visited_idxs, ext, body):
        lower_looplets(ctx, idx, visited_idxs, ext, body)
        return

    def default_loop(self, ctx, idx, ext, body):
        def assert_lowered(node):
            match node:
                case ntn.Access(_, _, (j, *_)):
                    if j == idx:
                        raise FinchCompileError(
                            node, f"Access with {j} should have been lowered already"
                        )
            return

        map(assert_lowered, PostOrderDFS(body))

        ctx_2 = ctx.scope()
        ctx_2.bindings[idx.name] = self.get_start(ext)
        ctx_2(body)
        return ctx_2.emit()


@dataclass(frozen=True)
class SimpleExtentFType:  # TODO: Remove once solved in Lookup looplet
    start: Any
    end: Any

    def get_start(self, ext):
        return self.start

    # def get_end(self, ext):
    #     return self.end

    @property
    def result_format(self):
        return self


class _CombineStyle(Enum):
    UNION = 1
    INTERSECT = 2


def _combine_extents(ext_1: Extent, ext_2: Extent, style: _CombineStyle) -> Extent:
    if style == _CombineStyle.UNION:
        start_fn, end_fn = min, max
    else:
        start_fn, end_fn = max, min

    start_1, start_2 = ext_1.ftype.get_start(ext_1), ext_2.ftype.get_start(ext_2)
    end_1, end_2 = ext_1.ftype.get_end(ext_1), ext_2.ftype.get_end(ext_2)
    return Extent(
        start=ntn.Call(ntn.Literal(start_fn), (start_1, end_1)),
        end=ntn.Call(ntn.Literal(end_fn), (start_2, end_2)),
    )


def intersect_extents(ext_1: Extent, ext_2: Extent) -> Extent:
    return _combine_extents(ext_1, ext_2, _CombineStyle.INTERSECT)


def union_extents(ext_1: Extent, ext_2: Extent) -> Extent:
    return _combine_extents(ext_1, ext_2, _CombineStyle.UNION)


def get_start(ext: Extent) -> Any:
    original_ext = ext
    if hasattr(ext, "result_format"):
        ext = ext.result_format
    elif hasattr(ext, "ftype"):
        ext = ext.ftype
    return ext.get_start(original_ext)


def get_end(ext: Extent) -> Any:
    original_ext = ext
    if hasattr(ext, "result_format"):
        ext = ext.result_format
    elif hasattr(ext, "ftype"):
        ext = ext.ftype
    return ext.get_end(original_ext)


def get_unit(ext: Extent) -> Any:
    original_ext = ext
    if hasattr(ext, "ftype"):
        ext = ext.ftype
    return ext.get_unit(original_ext)


def dimension(tns, mode: int) -> Extent:
    end = tns.shape[mode]
    return Extent(type(end)(0), end)


def numba_lower_dimension(ctx, tns, mode: int) -> str:
    return f"Numba_Extent(type({ctx(tns)}.shape[{mode}])(0), {ctx(tns)}.shape[{mode}])"


register_property(
    dimension,
    "__call__",
    "return_type",
    lambda op, x, y: ExtentFType(np.intp, np.intp),  # type: ignore[abstract]
)


register_property(
    dimension,
    "numba_literal",
    "__attr__",
    lambda func, ctx, tns, mode: numba_lower_dimension(ctx, tns, mode),
)
