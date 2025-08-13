from dataclasses import dataclass
from typing import FrozenSet, Iterable, Any
import numpy as np

import finch.finch_notation as ntn
from finch.compile import dimension
from finch.finch_notation.nodes import (
    Variable, Literal, Call, Access, Read, Update, Declare, Thaw, Freeze,
    Loop, If, Increment, Unwrap, Block, Slot, Unpack, Repack
)
from finch.finch_notation.interpreter import NotationInterpreter

import operator

from .tensor_def import TensorDef
from .tensor_stats import TensorStats


@dataclass(frozen=True)
class DC:
    from_indices: FrozenSet[str]
    to_indices: FrozenSet[str]
    value: float


class DCStats(TensorStats):
    def __init__(self, tensor: Any, fields: Iterable[str]):
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        self.fields = list(fields)
        self.tensor = np.asarray(tensor)
        self.dcs = self._calc_dc_from_structure()

    @classmethod
    def from_tensor(cls, tensor: Any, fields: Iterable[str]) -> None:
        return None

    def _calc_dc_from_structure(self) -> set[DC]:
        ndim = self.tensor.ndim
        if ndim == 1:
            return self._vector_structure_to_dcs()
        elif ndim == 2:
            return None
        else:
            raise NotImplementedError(f"DC analysis not implemented for {ndim}D tensors")

    def _vector_structure_to_dcs(self) -> set[DC]:
        """
        Build a Finch Notation program that counts non-fill entries in a 1-D tensor,
        execute it, and return a set of DC.
        """
        A  = ntn.Variable("A", np.ndarray)
        A_ = ntn.Slot("A_", np.ndarray)

        d  = ntn.Slot("d", np.ndarray)
        i  = ntn.Variable("i", np.int64)
        fill = ntn.Literal(self.tensordef.fill_value)

        prgm = ntn.Module((
            ntn.Function(
                ntn.Variable("count_nonfill", np.int64),
                (A,),
                ntn.Block((
                    ntn.Unpack(A_, A),
                    ntn.Declare(d, ntn.Literal(0.0), ntn.Literal(operator.add), ()),
                    ntn.Thaw(d, ntn.Literal(operator.add)),

                    ntn.Loop(
                        i,
                        ntn.Literal(int(self.tensor.shape[0])),
                        ntn.Block((
                            ntn.If(
                                ntn.Call(
                                    ntn.Literal(operator.ne),
                                    (
                                        ntn.Unwrap(ntn.Access(A_, ntn.Read(), (i,))),
                                        fill,
                                    ),
                                ),
                                ntn.Increment(
                                    ntn.Access(d, ntn.Update(ntn.Literal(operator.add)), ()),
                                    ntn.Literal(1.0),
                                ),
                            ),
                        )),
                    ),

                    ntn.Freeze(d, ntn.Literal(operator.add)),
                    ntn.Return(ntn.Unwrap(d)),
                    ntn.Repack(A_, A),
                )),
            ),
        ))

        mod = ntn.NotationInterpreter()(prgm)
        cnt = mod.count_nonfill(np.asarray(self.tensor))
        result = self.fields[0]

        return {DC(frozenset(), frozenset([result]), float(cnt))}

    @staticmethod
    def mapjoin(op, *args, **kwargs):
        pass

    @staticmethod
    def aggregate(op, dims, *args, **kwargs):
        pass

    @staticmethod
    def issimilar(*args, **kwargs):
        pass

    def estimate_non_fill_values(self):
        pass