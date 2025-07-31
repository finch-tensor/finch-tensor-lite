from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from finch.compile.lower import LoopletPass, SingletonExtentFormat

from .. import finch_assembly as asm
from .. import finch_notation as ntn
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
    head: Any
    split: Any
    tail: Any


@dataclass
class SequencePass(LoopletPass):
    @property
    def priority(self):
        return 0


@dataclass
class Run:
    body: Any

    @property
    def pass_request(self):
        return RunPass()


class RunPass(LoopletPass):
    @property
    def priority(self):
        return 0


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

    def __call__(self, ctx, idx, ext, body):
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
        ext_2 = SingletonExtentFormat.stack(idx_2)
        ctx_2(ext_2, body_2)
        start = ext.result_format.get_start(ext)
        stop = ext.result_format.get_end(ext)
        body_3 = ctx_2.emit()
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
        return 2

    def __call__(self, ctx, idx, ext, body):
        def leaf_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx and isinstance(tns, Leaf):
                        return ntn.Access(tns.body(ctx), mode, idxs)
            return None

        body_2 = PostWalk(leaf_node)(body)
        ctx.ctx(body_2)
