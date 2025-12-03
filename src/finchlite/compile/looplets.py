import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..compile.lower import LoopletPass, SimpleExtentFType, SingletonExtentFType
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


@dataclass
class SpikePass(LoopletPass):
    @property
    def priority(self):
        return 0


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
                asm.Literal(operator.add), (split_var.get(), asm.Literal(np.intp(1)))
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
                asm.Literal(operator.add), (split_var.get(), asm.Literal(np.intp(2)))
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

        # TODO: Proper Run Pass
        PostWalk(run_node)(body)
        ctx_2 = ctx.scope()
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
