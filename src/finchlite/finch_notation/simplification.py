from finchlite import finch_notation as ntn
from finchlite.algebra import is_annihilator
from finchlite.symbolic import Fixpoint, PostWalk, Rewrite
from finchlite.symbolic.stage import UnvalidatedForm

from .stages import NotationTransform


class LoopletSimplify(UnvalidatedForm, NotationTransform):
    def lower(self, term: ntn.Module) -> ntn.Module:
        return Rewrite(PostWalk(Fixpoint(lambda x: self.simplify(x))))(term)

    @staticmethod
    def simplify(term: ntn.NotationNode):
        from finchlite.compile import looplets as lplt
        from finchlite.tensor.scalar import Scalar

        match term:
            case ntn.Call(ntn.Literal(_) as op, args):
                for arg in args:
                    match arg:
                        case ntn.Unwrap(
                            ntn.Access(lplt.Run() as tns, ntn.Read(), idxs)
                        ) if isinstance(tns.body, Scalar) and is_annihilator(
                            op.val, tns.body.val
                        ):
                            return ntn.Unwrap(ntn.Access(tns, ntn.Read(), idxs))
        return None
