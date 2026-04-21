from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, reduce
from typing import Any

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import ffuncs
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

        cond = reduce(lambda x, y: asm.Call(asm.L(ffuncs.and_), (x, y)), conditions)

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

    def __call__(
        self,
        ctx: LoopletContext,
        idx: ntn.Variable,
        ext: SymbolicExtent,
        body,
    ) -> None:

        def stepper_body(ctx, node: ntn.NotationNode):
            match node:
                case ntn.Access(Stepper() as st, mode, (j, *idxs)) if j == idx:
                    return ntn.Access(
                        st.chunk,  # type: ignore[arg-type]
                        mode,
                        (j, *idxs),
                    )

        steppers: list[Stepper] = []

        for node in PostOrderDFS(body):
            match node:
                case Stepper() as st:
                    ctx.exec(st.seek(ctx, ext))
                    steppers.append(st)

        full_body = Rewrite(PostWalk(lambda node: stepper_body(ctx, node)))(body)

        final_cond = asm.Call(
            asm.L(ffuncs.lt),
            (
                asm.Call(
                    asm.L(ffuncs.min), tuple(ctx.ctx(s.stop(ctx)) for s in steppers)
                ),
                ctx.ctx(ext.get_end()),
            ),
        )

        stepper_blocks = []

        idx_start = ntn.Variable(ctx.freshen("i_start"), idx.result_type)
        ctx.exec(asm.Assign(ctx.ctx(idx_start), ctx.ctx(ext.get_start())))

        for stepper in steppers:
            cond = asm.Call(
                asm.L(ffuncs.eq),
                (
                    asm.Call(
                        asm.L(ffuncs.min), tuple(ctx.ctx(s.stop(ctx)) for s in steppers)
                    ),
                    ctx.ctx(stepper.stop(ctx)),
                ),
            )
            ctx_2 = ctx.scope()
            ext_2 = SymbolicExtent(
                idx_start,
                ntn.Call(ntn.L(ffuncs.add), (stepper.stop(ctx), ext.get_unit())),
            )
            ctx_2(ext_2, full_body)
            stepper_block = asm.If(
                cond,
                asm.Block(
                    (
                        *ctx_2.emit(),
                        asm.Assign(
                            ctx.ctx(idx_start),
                            asm.Call(
                                asm.L(ffuncs.add),
                                (ctx.ctx(stepper.stop(ctx)), ctx.ctx(ext_2.get_unit())),
                            ),
                        ),
                        stepper.next(ctx),
                        stepper.preamble(ctx),
                    )
                ),
            )
            stepper_blocks.append(stepper_block)

        ctx.exec(asm.WhileLoop(final_cond, asm.Block(tuple(stepper_blocks))))


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
                ntn.L(ffuncs.ge),
                (
                    ntn.Call(
                        ntn.L(ffuncs.sub),
                        (current_ext.get_end(), current_ext.get_unit()),
                    ),
                    remaining_ext.get_end(),
                ),
            )
        ):
            return Run(self.head)
        if prove(
            ntn.Call(
                ntn.L(ffuncs.eq),
                (current_ext.get_end(), remaining_ext.get_end()),
            )
        ):
            return self
        return Switch(
            asm.Call(
                asm.L(ffuncs.lt),
                (ctx.ctx(remaining_ext.get_end()), ctx.ctx(current_ext.get_end())),
            ),
            self,
            Run(self.head),
        )


class SequencePass(LoopletPass):
    """
    Lowers one Sequence looplet at the time.
    """

    @property
    def priority(self):
        return 4

    @classmethod
    def get_sequence_variations(
        cls, ctx, seqs: list[Sequence], heads, tails, ext: SymbolicExtent
    ) -> list[tuple]:
        ext_start, ext_end = ext.get_start(), ext.get_end()

        match seqs:
            case []:
                return [(heads, tails, ext_start, ext_end)]
            case [Sequence() as seq, *tail]:
                split = seq.split(ctx, ext)
                left = cls.get_sequence_variations(
                    ctx,
                    tail,
                    heads + [seq],
                    tails,
                    SymbolicExtent(
                        ext_start,
                        ntn.Call(ntn.L(ffuncs.min), (ext_end, split)),
                    ),
                )
                right = cls.get_sequence_variations(
                    ctx,
                    tail,
                    heads,
                    tails + [seq],
                    SymbolicExtent(
                        ntn.Call(ntn.L(ffuncs.max), (ext_start, split)),
                        ext_end,
                    ),
                )
                return left + right
            case other:
                raise Exception(f"Invalid sequence: {other}")

    def __call__(self, ctx: LoopletContext, idx, ext: SymbolicExtent, body):
        found_seqs: list[Sequence] = []

        for node in PostOrderDFS(body):
            match node:
                case Sequence() as seq:
                    found_seqs.append(seq)

        def sequence_node(
            node: ntn.NotationNode, heads: set[Sequence], tails: set[Sequence]
        ):
            match node:
                case ntn.Access(Sequence() as tns, mode, (j, *idxs)) if j == idx:
                    if tns in heads:
                        new_tns = tns.head(ctx, idx)  # type: ignore[call-arg]
                    elif tns in tails:
                        new_tns = tns.tail(ctx, idx)
                    else:
                        raise Exception(f"Seq: {tns} not present.")
                    return ntn.Access(new_tns, mode, (j, *idxs))

        variations = self.get_sequence_variations(ctx, found_seqs, [], [], ext)
        blocks = []

        for v in variations:
            heads, tails, subext_start, subext_end = v
            sub_body = PostWalk(partial(sequence_node, heads=heads, tails=tails))(body)
            subext = SymbolicExtent(subext_start, subext_end)
            sub_ctx = ctx.scope()
            sub_ctx(subext, sub_body)
            blocks += sub_ctx.emit()

        ctx.exec(asm.Block(tuple(blocks)))


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
