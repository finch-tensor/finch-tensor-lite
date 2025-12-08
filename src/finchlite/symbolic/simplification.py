from .. import finch_assembly as asm
from ..algebra import InitWrite, is_annihilator
from ..symbolic import Fixpoint, PostWalk, Rewrite


def simplify(ctx, prgm):
    from finchlite.interface.scalar import Scalar

    match prgm:
        case asm.Call(asm.Literal(_) as op, args):
            for arg in args:
                try:
                    match arg:
                        case asm.Literal(val) if isinstance(
                            val, Scalar
                        ) and is_annihilator(op.val, val.val):
                            return arg
                except AttributeError:
                    pass
            return None
        case asm.Block(
            (
                *_,
                asm.Store(
                    asm.Slot(_) as s1,
                    idx1,
                    asm.Call(
                        asm.Literal(InitWrite(val)),
                        (asm.Load(asm.Slot(_) as s2, idx2), asm.Literal(arg)),
                    ),
                ),
            )
        ) if s1 == s2 and idx1 == idx2 and val == arg.val:
            return asm.Block(())
        case asm.ForLoop(_, _, _, asm.Block(())):
            return asm.Block(())


def run_simplify(ctx, prgm):
    return Rewrite(PostWalk(Fixpoint(lambda x: simplify(ctx, x))))(prgm)
