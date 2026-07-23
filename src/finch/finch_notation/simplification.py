from finch import finch_notation as ntn
from finch.algebra import is_annihilator
from finch.symbolic import Fixpoint, PostWalk, Rewrite
from finch.symbolic.stage import UnvalidatedForm

from .stages import NotationTransform


class LoopletSimplify(UnvalidatedForm, NotationTransform):
    def lower(self, term: ntn.Module) -> ntn.Module:
        return Rewrite(PostWalk(Fixpoint(lambda x: self.simplify(x))))(term)

    @staticmethod
    def simplify(term: ntn.NotationNode):
        from finch.compile import looplets as lplt
        from finch.tensor.scalar import Scalar

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
