from .. import finch_notation as ntn
from ..algebra import is_annihilator
from ..symbolic import Fixpoint, PostWalk, Rewrite
from .stages import NotationTransform


class LoopletSimplify(NotationTransform):
    def __call__(self, term: ntn.Module) -> ntn.Module:
        return Rewrite(PostWalk(Fixpoint(lambda x: self.simplify(x))))(term)

    @classmethod
    def simplify(cls, term: ntn.NotationNode):
        from ..compile import looplets as lplt
        from ..interface.scalar import Scalar

        match term:
            case ntn.Call(ntn.Literal(_) as op, args):
                for arg in args:
                    match arg:
                        case ntn.Unwrap(
                            ntn.Access(lplt.Run() as tns, ntn.Read(), idxs)
                        ):
                            if isinstance(tns.body, Scalar) and is_annihilator(
                                op.val, tns.body.val
                            ):
                                return ntn.Unwrap(ntn.Access(tns, ntn.Read(), idxs))
        return None
