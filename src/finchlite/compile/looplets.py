import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..compile.extents import Extent, SimpleExtentFType, SingletonExtentFType, get_end, get_start, get_unit
from ..compile.lower import LoopletPass
from ..finch_notation.proves import prove
from ..symbolic import PostWalk


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
        return 0


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
        return 0


@dataclass
class Stepper:
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
        return StepperPass()


class StepperPass(LoopletPass):
    @property
    def priority(self):
        return 0


@dataclass
class Spike:
    body: Any
    tail: Any

    @property
    def pass_request(self):
        return SpikePass()

    def truncate(self, ext_1, ext_2):
        if prove(
            ntn.Call(
                ntn.Literal(operator.ge),
                (
                    ntn.Call(
                        ntn.Literal(operator.sub), (get_end(ext_1), get_unit(ext_1))
                    ),
                    get_end(ext_2),
                ),
            )
        ):
            return Run(self.body)
        if prove(ntn.Call(ntn.Literal(operator.eq), (get_end(ext_1), get_end(ext_2)))):
            return self
        raise NotImplementedError


@dataclass
class SpikePass(LoopletPass):
    @property
    def priority(self):
        return 0

    def __call__(self, ctx, idx, ext, body):
        def spike_body(node):
            match node:
                case ntn.Access(Spike(body, _), mode, (j, *idxs)) if j == idx:
                    return ntn.Access(body(ctx), mode, (j, *idxs))

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
                    return ntn.Access(tail(ctx), mode, (j, *idxs))

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
        return 1

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
        ext_2 = SimpleExtentFType(
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
        ext_3 = SimpleExtentFType(
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
        return 1

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
        return 0


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
        return 0

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
        start = ext.result_format.get_start(ext)  # TODO: We should accept arbitrary ext
        stop = ext.result_format.get_end(ext)
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
