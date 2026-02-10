import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, reduce
from operator import eq, ge, le
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..compile.extents import (
    Extent,
    ExtentFType,
    SingletonExtentFType,
    get_end,
    get_start,
    get_unit,
    intersect_extents,
)
from ..compile.lower import LoopletContext, LoopletPass
from ..finch_notation.proves import prove
from ..symbolic import PostOrderDFS, PostWalk, Rewrite


class Looplet(ABC):
    @property
    @abstractmethod
    def pass_request(self): ...


@dataclass
class Thunk:
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

    def __call__(self, ctx: LoopletContext, idx, visited_idxs, ext, body):
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


@dataclass
class Switch:
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

    def __call__(self, ctx: LoopletContext, idx, visited_idxs, ext, body):
        conditions: list[ntn.Call] = []

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

        cond = reduce(lambda x, y: ntn.Call(ntn.L(operator.and_), (x, y)), conditions)

        body_else = PostWalk(switch_node_else)(body)
        ctx_3 = ctx.scope()
        ctx_3(ext, body_else)

        ctx.exec(asm.Block((asm.If(cond, body_if, body_else),)))


@dataclass
class Stepper:
    preamble: Any = None
    stop: Callable = lambda ctx, ext: None
    chunk: Looplet = None
    next: Callable[[Any, Any], Any] = lambda ctx, ext: None
    body: Callable = lambda ctx, ext: None
    seek: Callable = lambda ctx, start: (_ for _ in ()).throw(
        NotImplementedError("seek not implemented error")
    )

    @property
    def pass_request(self):
        return StepperPass()


@dataclass
class StepperPass(LoopletPass):
    count: int = 1

    @property
    def priority(self):
        return 3

    def combine_with(self, other):
        if isinstance(other, StepperPass):
            return StepperPass(self.count + other.count)
        return max(self, other)

    def stepper_range(self, ctx, node: Stepper, ext: ExtentFType):
        if (preamble := node.preamble) is not None:
            ctx.exec(preamble)
        ext_2 = ExtentFType(get_start(ext), node.stop(ctx, ext))
        return ext_2.bound_below(get_unit(ext))

    def stepper_body(self, ctx, node: Stepper, ext, remaining_ext: ExtentFType):
        next = node.next(ctx, remaining_ext)
        assert isinstance(node.chunk, Spike)
        full_chunk = Thunk(
            body=lambda ctx, ext: node.chunk,
            epilogue=lambda ctx, ext: next,
        )
        truncated_chunk = node.chunk.truncate(
            ext,
            remaining_ext.bound_above(
                ntn.Call(ntn.L(operator.sub), (get_end(ext), get_unit(ext)))
            ),
        )

        if prove(ntn.Call(ntn.L(le), (node.stop(ctx, ext), get_end(remaining_ext)))):
            return full_chunk
        if prove(ntn.Call(ntn.L(ge), (node.stop(ctx, ext), get_end(remaining_ext)))):
            return truncated_chunk
        return Switch(
            asm.Call(asm.L(eq), (node.stop(ctx, ext), get_end(remaining_ext))),
            full_chunk,
            truncated_chunk,
        )

    def __call__(self, ctx: LoopletContext, idx, visited_idxs, ext, body):
        i0 = asm.Variable(ctx.freshen(idx.name, "_start"), idx.result_format)

        def stepper_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Stepper):
                        tns_2 = Thunk(
                            epilogue=tns.next(tns.pos), body=tns.body(ctx, idx_2)
                        )
                        return ntn.Access(tns_2, mode, (j, *idxs))
            return None

        def stepper_body(ctx, node: ntn.NotationNode, ext, ext_2):
            match node:
                case ntn.Access(Stepper() as st, mode, (j, *idxs)) if j == idx:
                    return ntn.Access(
                        self.stepper_body(ctx, st, ext, ext_2), mode, (j, idxs)
                    )

        ctx_2 = ctx.scope()

        # current remaninig extent
        ext_1 = ext.bound_below(get_unit(ext))

        steppers: list[Stepper] = []

        for node in PostOrderDFS(body):
            match node:
                case Stepper() as st:
                    ctx.exec(st.seek(ctx, ext))
                    steppers.append(st)

        # intersection of all steppers
        ext_2 = reduce(
            lambda x, y: intersect_extents(x, y),
            [self.stepper_range(ctx, st, ext_1) for st in steppers],
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

        if not prove(
            ntn.Call(ntn.Literal(operator.ge), (ext_3.get_measure(), ext_3.get_unit()))
        ):
            truncated_body = asm.If(
                asm.Call(asm.Literal(ge), (ext_3.get_end(), ext_3.get_start())),
                truncated_body,
            )

        while_loop_body = asm.IfElse(
            asm.Call(asm.Literal(operator.lt), (ext_2.get_stop(), ext.get_stop())),
            ctx_full_body.emit(),
            ctx_truncated_body.emit(),
        )

        ctx_2(ext, while_loop_body)
        while_loop_block = asm.Block(ctx_2.emit())
        ctx.exec(asm.WhileLoop(asm.Literal(np.True_), while_loop_block))


@dataclass
class Spike:
    body: Any
    tail: Any

    @property
    def pass_request(self):
        return SpikePass()

    def truncate(self, current_ext, remaining_ext):
        if prove(
            ntn.Call(
                ntn.Literal(operator.ge),
                (
                    ntn.Call(
                        ntn.Literal(operator.sub),
                        (get_end(current_ext), get_unit(current_ext)),
                    ),
                    get_end(remaining_ext),
                ),
            )
        ):
            return Run(self.body)
        if prove(
            ntn.Call(
                ntn.Literal(operator.eq), (get_end(current_ext), get_end(remaining_ext))
            )
        ):
            return self
        return Switch(
            asm.Call(
                asm.Literal(operator.lt), (get_end(remaining_ext), get_end(current_ext))
            ),
            Run(self.body),
            self,
        )


@dataclass
class SpikePass(LoopletPass):
    @property
    def priority(self):
        return 5

    def __call__(self, ctx, idx, visited_idxs, ext, body):
        def spike_body(node):
            match node:
                case ntn.Access(Spike(body, _), mode, (j, *idxs)) if j == idx:
                    return ntn.Access(body(ctx, idx), mode, (j, *idxs))

        body_body = PostWalk(spike_body)(body)
        body_ctx = ctx.scope()
        body_ext = Extent(
            get_start(ext),
            asm.Call(asm.Literal(operator.sub), (get_end(ext), get_unit(ext))),
        )
        body_ctx(body_ext, body_body)

        def spike_tail(node):
            match node:
                case ntn.Access(Spike(_, tail), mode, (j, *idxs)) if j == idx:
                    return ntn.Access(tail(ctx, idx), mode, (j, *idxs))

        tail_body = PostWalk(spike_tail)(body)
        tail_ctx = ctx.scope()
        tail_ext = SingletonExtentFType.stack(get_end(ext))
        tail_ctx(tail_ext, tail_body)

        ctx.exec(asm.Block((*body_ctx.emit(), *tail_ctx.emit())))


@dataclass
class Sequence:
    head: Callable
    split: Callable
    tail: Callable

    @property
    def pass_request(self):
        return SequencePass()


@dataclass
class SetOnce:
    obj: Any | None = None

    def set(self, obj):
        if self.obj is None:
            self.obj = obj
        elif self.obj != obj:
            raise Exception(f"SetOnce object already set with: {self.obj}")

    def get(self):
        if self.obj is None:
            raise Exception("The SetOnce object hasn't been set.")
        return self.obj


@dataclass
class SequencePass(LoopletPass):
    @property
    def priority(self):
        return 4

    def __call__(self, ctx, idx, visited_idxs, ext, body):
        split_var = SetOnce()

        def sequence_node(node, seq_field: str):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Sequence):
                        split_var.set(tns.split(ctx, idx, visited_idxs))
                        return ntn.Access(
                            getattr(tns, seq_field)(ctx, idx), mode, (j, *idxs)
                        )
            return None

        # process head
        sequence_head = partial(sequence_node, seq_field="head")
        body_head = PostWalk(sequence_head)(body)
        ext_2 = ExtentFType(
            ext.result_format.get_start(ext),
            asm.Call(
                asm.Literal(min),
                (
                    asm.Call(
                        asm.Literal(operator.add),
                        (split_var.get(), asm.Literal(np.intp(1))),
                    ),
                    ext.result_format.get_end(ext),
                ),
            ),
        )
        ctx_2 = ctx.scope()
        ctx_2(ext_2, body_head)
        emitted_head = ctx_2.emit()

        # process tail
        sequence_tail = partial(sequence_node, seq_field="tail")
        body_tail = PostWalk(sequence_tail)(body)
        ext_3 = ExtentFType(
            asm.Call(
                asm.Literal(operator.add), (split_var.get(), asm.Literal(np.intp(1)))
            ),
            ext.result_format.get_end(ext),
        )
        ctx_3 = ctx.scope()
        ctx_3(ext_3, body_tail)
        emitted_tail = ctx_3.emit()

        ctx.exec(asm.Block((*emitted_head, *emitted_tail)))


@dataclass
class Run:
    body: Any

    @property
    def pass_request(self):
        return RunPass()


class RunPass(LoopletPass):
    @property
    def priority(self):
        return 6

    def __call__(self, ctx, idx, visited_idxs, ext, body):
        def run_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Run):
                        return ntn.Access(tns.body(ctx), mode, (j, *idxs))
            return None

        body_2 = PostWalk(run_node)(body)
        ctx_2 = ctx.scope()
        ctx_2(ext, body_2)
        ctx.exec(asm.Block(ctx_2.emit()))


@dataclass
class AcceptRun:
    body: Any

    @property
    def pass_request(self):
        return AcceptRunPass()


class AcceptRunPass(LoopletPass):
    @property
    def priority(self):
        return 1


@dataclass
class Null:
    pass


@dataclass
class Lookup:
    body: Callable

    @property
    def pass_request(self):
        return LookupPass()


class LookupPass(LoopletPass):
    @property
    def priority(self):
        return 2

    def __call__(self, ctx, idx, visited_idxs, ext, body):
        idx_2 = asm.Variable(ctx.freshen(idx.name), idx.result_format)

        def lookup_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Lookup):
                        tns_2 = tns.body(
                            ctx,
                            idx_2,
                        )
                        return ntn.Access(tns_2, mode, (j, *idxs))
            return None

        body_2 = PostWalk(lookup_node)(body)
        ctx_2 = ctx.scope()
        ext_2 = SingletonExtentFType.stack(idx_2)
        ctx_2(ext_2, body_2)
        start = get_start(ext)
        stop = get_end(ext)
        # print(ext)
        # start = ext.result_format.get_start(ext)  # TODO: We should accept arbitrary ext
        # stop = ext.result_format.get_end(ext)
        body_3 = asm.Block(ctx_2.emit())
        ctx.exec(asm.ForLoop(idx_2, start, stop, body_3))


@dataclass
class Jumper:
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
class Leaf:
    body: Callable

    @property
    def pass_request(self):
        return LeafPass()


class LeafPass(LoopletPass):
    @property
    def priority(self):
        return 0

    def __call__(self, ctx, idx, visited_idxs, ext, body):
        def leaf_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Leaf):
                        return ntn.Access(tns.body(ctx), mode, idxs)
            return None

        body_2 = PostWalk(leaf_node)(body)
        ctx.ctx(body_2, visited_idxs + (idx,))  # calling NotationContext
