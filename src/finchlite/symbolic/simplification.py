from .. import finch_assembly as asm
from ..algebra import InitWrite, is_annihilator, is_identity
from ..symbolic import Fixpoint, PostWalk, Rewrite


def simplify(ctx, prgm):
    from finchlite.interface.scalar import Scalar

    match prgm:
        # op(..., arg, ...) where arg is anihilator => arg
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
        # slot(a, idx) = op(slot(a, idx), arg) where RHS is:
        #   1. InitWrite(x)(slot(a, idx), x)
        #   2. op(slot(a, idx), arg) and arg is an identity for op
        # is removed
        case asm.Block(
            (
                *_,
                asm.Store(
                    asm.Slot(_) as s1,
                    idx1,
                    asm.Call(
                        asm.Literal(op),
                        (asm.Load(asm.Slot(_) as s2, idx2), asm.Literal(arg)),
                    ),
                ),
            )
        ) if s1 == s2 and idx1 == idx2:
            if isinstance(op, InitWrite) and op.value == arg.val:
                return asm.Block(())
            if is_identity(op, arg.val):
                return asm.Block(())
        # loop(...) {} is removed
        case asm.ForLoop(_, _, _, asm.Block(())):
            return asm.Block(())
    return None


def run_simplify(ctx, prgm):
    return Rewrite(PostWalk(Fixpoint(lambda x: simplify(ctx, x))))(prgm)
