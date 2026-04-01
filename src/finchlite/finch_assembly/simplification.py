# AI modified: 2026-04-01T17:18:51Z 0de216cc18e91710a9b1a0328f5b181137d8901b
# AI modified: 2026-04-01T17:28:42Z 0de216cc18e91710a9b1a0328f5b181137d8901b
from .. import finch_assembly as asm
from ..algebra import ffunc, is_annihilator, is_identity, overwrite
from ..algebra.algebra import FinchOperator
from ..symbolic import Fixpoint, PostWalk, Rewrite
from .stages import AssemblyTransform


class AssemblySimplify(AssemblyTransform):
    def __call__(self, term: asm.Module) -> asm.Module:
        return Rewrite(PostWalk(Fixpoint(lambda x: self.simplify(x))))(term)

    @classmethod
    def simplify(cls, term: asm.AssemblyNode):
        from finchlite.interface.scalar import Scalar

        match term:
            # overwrite(x, y) => y
            case asm.Call(asm.Literal(fn), (_, y)) if fn is overwrite:
                return y
            # op(..., arg, ...) where arg is anihilator => arg
            case asm.Call(asm.Literal(_) as op, args):
                assert isinstance(op.val, FinchOperator)
                for arg in args:
                    match arg:
                        case asm.Literal(val) if isinstance(
                            val, Scalar
                        ) and is_annihilator(op.val, val.val):
                            return arg
                return None
            # slot(a, idx) = op(slot(a, idx), arg) where RHS is:
            #   1. init_write(x)(slot(a, idx), x)
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
                if op == ffunc.init_write(arg.val):
                    return asm.Block(())
                assert isinstance(op, FinchOperator)
                if is_identity(op, arg.val):
                    return asm.Block(())
            # loop(...) {} is removed
            case asm.ForLoop(_, _, _, asm.Block(())):
                return asm.Block(())
            # if(...) {} is removed
            case asm.If(_, asm.Block(())):
                return asm.Block(())
            # block(..., block(), ...) => block(...)
            case asm.Block(bodies):
                for i, b in enumerate(bodies):
                    match b:
                        case asm.Block(()):
                            return asm.Block((*bodies[:i], *bodies[i + 1 :]))
        return None
