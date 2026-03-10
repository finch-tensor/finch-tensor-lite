from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from operator import add, and_, eq, ge, le, lt, sub
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..compile.extents import intersect_extents
from ..compile.lower import (
    LoopletContext,
    LoopletPass,
    SymbolicExtent,
)
from ..finch_notation.proves import prove
from ..symbolic import PostOrderDFS, PostWalk, Rewrite


class Looplet(ABC):
    @property
    @abstractmethod
    def pass_request(self): ...


@dataclass
class Thunk(Looplet):
    preamble: Any = None
    body: Any = None
    epilogue: Any = None

    @property
    def pass_request(self):
        return ThunkPass()


class ThunkPass(LoopletPass):
    @property
    def priority(self):
        return 8

    def __call__(self, ctx: LoopletContext, idx, ext, body):
        def thunk_body(ctx, node: ntn.NotationNode):
            match node:
                case ntn.Access(Thunk() as thnk, mode, (j, *idxs)) if j == idx:
                    if (preamble := thnk.preamble) is not None:
                        ctx.exec(preamble(ctx, idx))
                    if (epilogue := thnk.epilogue) is not None:
                        ctx.post(epilogue(ctx, idx))
                    return ntn.Access(thnk.body(ctx, ext), mode, (j, *idxs))

        ctx_2 = ctx.scope()
        body = Rewrite(PostWalk(lambda x: thunk_body(ctx_2, x)))(body)
        ctx_2(ext, body)
        ctx.exec(asm.Block(ctx_2.emit()))


@dataclass
class Switch(Looplet):
    cond: Any
    if_true: Any
    if_false: Any

    @property
    def pass_request(self):
        return SwitchPass()


class SwitchPass(LoopletPass):
    @property
    def priority(self):
        return 7

    def __call__(self, ctx: LoopletContext, idx, ext: SymbolicExtent, body):
        conditions: list[asm.Call] = []

        def switch_node_if(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Switch):
                        conditions.append(tns.cond)
                        return ntn.Access(tns.if_true, mode, (j, *idxs))
            return None

        def switch_node_else(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Switch):
                        return ntn.Access(tns.if_false, mode, (j, *idxs))
            return None

        body_if = PostWalk(switch_node_if)(body)
        ctx_2 = ctx.scope()
        ctx_2(ext, body_if)

        cond = reduce(lambda x, y: asm.Call(asm.L(and_), (x, y)), conditions)

        body_else = PostWalk(switch_node_else)(body)
        ctx_3 = ctx.scope()
        ctx_3(ext, body_else)

        ctx.exec(
            asm.Block(
                (asm.IfElse(cond, asm.Block(ctx_2.emit()), asm.Block(ctx_3.emit())),)
            )
        )


@dataclass
class Stepper(Looplet):
    preamble: Any = None
    stop: Callable = lambda ctx, ext: None
    chunk: Looplet | None = None
    next: Callable[[Any], Any] = lambda ctx: None
    body: Callable = lambda ctx, ext: None
    seek: Callable = lambda ctx, start: (_ for _ in ()).throw(
        NotImplementedError("seek not implemented error")
    )

    @property
    def pass_request(self):
        return StepperPass()


class StepperPass(LoopletPass):
    count: int = 1

    @property
    def priority(self):
        return 3

    def combine_with(self, other):
        if isinstance(other, StepperPass):
            return StepperPass(self.count + other.count)
        return max(self, other)

    def stepper_range(
        self,
        ctx: LoopletContext,
        node: Stepper,
        ext: SymbolicExtent,
        idx_start: ntn.Variable,
    ) -> SymbolicExtent:
        if (preamble := node.preamble) is not None:
            ctx.exec(preamble(ctx))
        return SymbolicExtent(idx_start, node.stop(ctx))

    def stepper_body(
        self,
        ctx: LoopletContext,
        node: Stepper,
        total_ext: SymbolicExtent,
        chunk_ext: SymbolicExtent,
    ):
        next = node.next(ctx)
        assert isinstance(node.chunk, Sequence)
        full_chunk = Thunk(
            body=lambda ctx, ext: node.chunk,
            epilogue=lambda ctx, ext: next,
        )
        truncated_chunk = node.chunk.truncate(
            ctx,
            total_ext,
            chunk_ext.bound_above(
                ntn.Call(ntn.L(sub), (total_ext.get_end(), total_ext.get_unit()))
            ),
        )

        if prove(ntn.Call(ntn.L(le), (node.stop(ctx), chunk_ext.get_end()))):
            return full_chunk
        if prove(ntn.Call(ntn.L(ge), (node.stop(ctx), chunk_ext.get_end()))):
            return truncated_chunk
        return Switch(
            asm.Call(
                asm.L(eq),
                (ctx.ctx(node.stop(ctx)), ctx.ctx(chunk_ext.get_end())),
            ),
            full_chunk,
            truncated_chunk,
        )

    def __call__(
        self,
        ctx: LoopletContext,
        idx: ntn.Variable,
        ext: SymbolicExtent,
        body,
    ) -> None:

        def stepper_body(ctx, node: ntn.NotationNode, total_ext, chunk_ext):
            match node:
                case ntn.Access(Stepper() as st, mode, (j, *idxs)) if j == idx:
                    return ntn.Access(
                        self.stepper_body(ctx, st, total_ext, chunk_ext),
                        mode,
                        (j, *idxs),
                    )

        # total remaining extent
        ext_1 = ext.bound_below(ext.get_unit())

        steppers: list[Stepper] = []

        for node in PostOrderDFS(body):
            match node:
                case Stepper() as st:
                    ctx.exec(st.seek(ctx, ext))
                    steppers.append(st)

        ctx_2 = ctx.scope()

        idx_start = ntn.Variable(ctx.freshen("i_start"), idx.result_format)
        ctx.exec(asm.Assign(ctx.ctx(idx_start), ctx.ctx(ext.get_start())))

        # intersection of all steppers - chunk extent
        ext_2 = reduce(
            lambda x, y: intersect_extents(x, y),
            [self.stepper_range(ctx_2, st, ext_1, idx_start) for st in steppers],
        )

        ctx_full_body = ctx_2.scope()
        full_body = Rewrite(
            PostWalk(lambda node: stepper_body(ctx_2, node, ext_1, ext_2))
        )(body)
        ctx_full_body(ext_2, full_body)

        ext_3 = intersect_extents(ext_1, ext_2)

        ctx_truncated_body = ctx_2.scope()
        truncated_body = Rewrite(
            PostWalk(lambda node: stepper_body(ctx_truncated_body, node, ext_1, ext_3))
        )(body)
        ctx_truncated_body(ext_3, truncated_body)

        if not prove(ntn.Call(ntn.L(ge), (ext_3.get_measure(), ext_3.get_unit()))):
            truncated_body = asm.If(
                asm.Call(
                    asm.L(ge),
                    (ctx.ctx(ext_3.get_end()), ctx.ctx(ext_3.get_start())),
                ),
                truncated_body,
            )

        while_loop_body = asm.IfElse(
            asm.Call(
                asm.L(lt),
                (
                    ctx.ctx(ext_2.get_end()),
                    asm.Call(
                        asm.L(sub),
                        (ctx.ctx(ext.get_end()), ctx.ctx(ext.get_unit())),
                    ),
                ),
            ),
            asm.Block(ctx_full_body.emit()),
            asm.Block((*ctx_truncated_body.emit(), asm.Break())),
        )
        while_loop_block = asm.Block(
            (
                *ctx_2.emit(),
                while_loop_body,
                asm.Assign(
                    ctx.ctx(idx_start),
                    asm.Call(
                        asm.L(add),
                        (ctx.ctx(ext_2.get_end()), ctx.ctx(ext_2.get_unit())),
                    ),
                ),
            )
        )
        ctx.exec(asm.WhileLoop(asm.L(np.True_), while_loop_block))


@dataclass
class Sequence(Looplet):
    head: Callable
    split: Callable
    tail: Callable

    @property
    def pass_request(self):
        return SequencePass()

    def truncate(
        self,
        ctx: LoopletContext,
        current_ext: SymbolicExtent,
        remaining_ext: SymbolicExtent,
    ):
        if prove(
            ntn.Call(
                ntn.L(ge),
                (
                    ntn.Call(
                        ntn.L(sub),
                        (current_ext.get_end(), current_ext.get_unit()),
                    ),
                    remaining_ext.get_end(),
                ),
            )
        ):
            return Run(self.head)
        if prove(
            ntn.Call(
                ntn.L(eq),
                (current_ext.get_end(), remaining_ext.get_end()),
            )
        ):
            return self
        return Switch(
            asm.Call(
                asm.L(lt),
                (ctx.ctx(remaining_ext.get_end()), ctx.ctx(current_ext.get_end())),
            ),
            self,
            Run(self.head),
        )


@dataclass
class SequencePass(LoopletPass):
    """
    Lowers one Sequence looplet at the time.
    """

    @property
    def priority(self):
        return 4

    def __call__(self, ctx: LoopletContext, idx, ext: SymbolicExtent, body):
        found_sequence: Sequence | None = None

        # process head
        def sequence_head(node: ntn.NotationNode):
            match node:
                case ntn.Access(Sequence() as tns, mode, (j, *idxs)) if j == idx:
                    nonlocal found_sequence
                    found_sequence = tns
                    return ntn.Access(tns.head(ctx, idx), mode, (j, *idxs))  # type: ignore[call-arg]

        body_head = PostWalk(sequence_head)(body)
        assert isinstance(found_sequence, Sequence)
        split_var = found_sequence.split(ctx, ext)
        ext_2 = SymbolicExtent(
            ext.get_start(),
            ntn.Call(ntn.Literal(min), (split_var, ext.get_end())),
        )
        ctx_2 = ctx.scope()
        ctx_2(ext_2, body_head)
        emitted_head = ctx_2.emit()

        # process tail
        def sequence_tail(node: ntn.NotationNode):
            match node:
                case ntn.Access(Sequence() as tns, mode, (j, *idxs)) if (
                    # Reported: https://github.com/python/mypy/issues/20904
                    j == idx and tns is found_sequence  # type: ignore[has-type]
                ):
                    return ntn.Access(tns.tail(ctx, idx), mode, (j, *idxs))

        body_tail = PostWalk(sequence_tail)(body)
        ext_3 = SymbolicExtent(split_var, ext.get_end())
        ctx_3 = ctx.scope()
        ctx_3(ext_3, body_tail)
        emitted_tail = ctx_3.emit()

        ctx.exec(asm.Block((*emitted_head, *emitted_tail)))


@dataclass
class Run(Looplet):
    body: Any

    @property
    def pass_request(self):
        return RunPass()


class RunPass(LoopletPass):
    @property
    def priority(self):
        return 6

    def __call__(self, ctx, idx, ext, body):
        def run_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Run):
                        return ntn.Access(tns.body(ctx, idx), mode, (j, *idxs))
            return None

        body_2 = PostWalk(run_node)(body)
        ctx_2 = ctx.scope()
        ctx_2(ext, body_2)
        ctx.exec(asm.Block(ctx_2.emit()))


@dataclass
class AcceptRun(Looplet):
    body: Any

    @property
    def pass_request(self):
        return AcceptRunPass()


class AcceptRunPass(LoopletPass):
    @property
    def priority(self):
        return 1


@dataclass
class Lookup(Looplet):
    body: Callable

    @property
    def pass_request(self):
        return LookupPass()


class LookupPass(LoopletPass):
    @property
    def priority(self):
        return 2

    def __call__(self, ctx: LoopletContext, idx, ext: SymbolicExtent, body):
        def lookup_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Lookup):
                        tns_2 = tns.body(
                            ctx,
                            idx,
                        )
                        return ntn.Access(tns_2, mode, (j, *idxs))
            return None

        body_2 = PostWalk(lookup_node)(body)
        ctx_2 = ctx.scope()
        ext_2 = SymbolicExtent.point(idx)
        ctx_2(ext_2, body_2)
        body_3 = asm.Block(ctx_2.emit())

        if ext.is_sym_point():
            ctx.exec(
                asm.Block(
                    (asm.Assign(ctx.ctx(idx), ctx.ctx(ext.get_start())), *body_3.bodies)
                )
            )
        else:
            ctx.exec(
                asm.ForLoop(
                    ctx.ctx(idx),
                    ctx.ctx(ext.get_start()),
                    ctx.ctx(ext.get_end()),
                    body_3,
                )
            )


@dataclass
class Jumper(Looplet):
    preamble: Any = None
    stop: Callable = lambda ctx, ext: None
    chunk: Any = None
    next: Callable = lambda ctx, ext: None
    body: Callable = lambda ctx, ext: None
    seek: Callable = lambda ctx, start: (_ for _ in ()).throw(
        NotImplementedError("seek not implemented error")
    )

    @property
    def pass_request(self):
        return JumperPass()


class JumperPass(LoopletPass):
    @property
    def priority(self):
        return 0


@dataclass
class Leaf(Looplet):
    body: Callable

    @property
    def pass_request(self):
        return LeafPass()


class LeafPass(LoopletPass):
    @property
    def priority(self):
        return 0

    def __call__(self, ctx, idx, ext, body):
        def leaf_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Leaf):
                        return ntn.Access(tns.body(ctx), mode, tuple(idxs))
            return None

        body_2 = PostWalk(leaf_node)(body)
        ctx.ctx(body_2)  # calling AssemblyContext
