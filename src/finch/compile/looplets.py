from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Thunk:
    preamble: Any = None
    body: Any = None
    epilogue: Any = None


@dataclass
class Switch:
    cases: Any


@dataclass
class Case:
    cond: Any
    body: Any


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


@dataclass
class Spike:
    body: Any
    tail: Any


@dataclass
class Sequence:
    phases: Any


@dataclass
class Run:
    body: Any


@dataclass
class AcceptRun:
    body: Any


@dataclass
class Phase:
    body: Any
    start: Callable = lambda ctx, ext: None
    stop: Callable = lambda ctx, ext: None
    range: Callable = lambda ctx, ext: None


@dataclass
class Null:
    pass


@dataclass
class Lookup:
    body: Callable

    def get_body(self, ctx, idx):
        return self.body(ctx, idx)


class LookupPass:
    def __init__(self, body: Any):
        self.body = body

    def __call__(self, ctx, ext):
        idx = asm.Variable(ctx.freshen(ctx.idx, "_lookup"))

        def get_body(node):
            match node:
                case ntn.Access(tns, _, (j, *_)):
                    if j == self.ctx.idx:
                        return tns.fmt.get_body(self.ctx, idx)
            return None

        body_2 = self.ctx(Rewrite(PostWalk(get_body))(self.body))
        return asm.ForLoop(
            start=ext.start,
            stop=ext.stop,
            step=1,
            body=body_2,
        )


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


@dataclass
class FillLeaf:
    body: Any


@dataclass
class ShortCircuitVisitor:
    ctx: Any
