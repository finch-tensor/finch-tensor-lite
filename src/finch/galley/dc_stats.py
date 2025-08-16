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
        if ndim == 2:
            return self._matrix_structure_to_dcs()
        else:
            raise NotImplementedError(f"DC analysis not implemented for {ndim}D tensors")

    def _vector_structure_to_dcs(self) -> set[DC]:
        """
        Build a Finch Notation program that counts non-fill entries in a 1-D tensor,
        execute it, and return a set of DC.
        """
        A = ntn.Variable("A", np.ndarray)
        A_ = ntn.Slot("A_", np.ndarray)

        d = ntn.Variable("d", np.int64)
        i = ntn.Variable("i", np.int64)
        m = ntn.Variable("m", np.int64)

        prgm = ntn.Module(
            (
                ntn.Function(
                    ntn.Variable("vector_structure_to_dcs", np.int64),
                    (A,),
                    ntn.Block(
                        (
                            ntn.Assign(
                                m, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0)))
                            ),
                            ntn.Assign(d, ntn.Literal(np.int64(0))),
                            ntn.Unpack(A_, A),
                            ntn.Loop(
                                i,
                                m,
                                ntn.Assign(
                                    d,
                                    ntn.Call(
                                        Literal(operator.add),
                                        (
                                            d,
                                            ntn.Unwrap(ntn.Access(A_, ntn.Read(), (i,))),
                                        ),
                                    ),
                                ),
                            ),
                            ntn.Repack(A_, A),
                            ntn.Return(d),
                        )
                    ),
                ),
            )
        )

        mod = ntn.NotationInterpreter()(prgm)
        cnt = mod.vector_structure_to_dcs(self.tensor)
        result = self.fields[0]

        return {DC(frozenset(), frozenset([result]), float(cnt))}

    def _matrix_structure_to_dcs(self) -> set[DC]:

        A = ntn.Variable("A", np.ndarray)
        A_ = ntn.Slot("A_", np.ndarray)

        i = ntn.Variable("i", np.int64)
        j = ntn.Variable("j", np.int64)
        ni = ntn.Variable("ni", np.int64)
        nj = ntn.Variable("nj", np.int64)

        dij = ntn.Variable("dij", np.int64)

        X = ntn.Slot("X", np.ndarray)
        Y = ntn.Slot("Y", np.ndarray)
        xi = ntn.Variable("xi", np.int64)
        yj = ntn.Variable("yj", np.int64)

        d_i    = ntn.Variable("d_i",   np.float64)
        d_i_j  = ntn.Variable("d_i_j", np.float64)
        d_j    = ntn.Variable("d_j",   np.float64)
        d_j_i  = ntn.Variable("d_j_i", np.float64)

        prgm = ntn.Module(
            ntn.Function(
                ntn.Variable("total_nnz", np.int64),
                (A,),
                ntn.Block(
                    ntn.Assign(
                        ni, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0)))
                    ),
                    ntn.Assign(
                        nj, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1)))
                    ),
                    ntn.Assign(
                        dij, ntn.Literal(np.int64(0))
                    ),
                    ntn.Unpack(A_,A),
                    ntn.Loop(
                        i,
                        ni,
                        ntn.Loop(
                            j,
                            nj,
                            ntn.Assign(
                                dij,
                                ntn.Call(
                                    ntn.Literal(operator.add),
                                    (
                                        dij,
                                        ntn.Unwrap(
                                            ntn.Access(
                                                A_,
                                                ntn.Read(),
                                                (j, i)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    ntn.Repack(A_,A),
                    ntn.Return(dij),
                )
            ),
            ntn.Function(
                ntn.Variable("matrix_structure_to_dcs", tuple),
                (A,),
                ntn.Block(
                    ntn.Assign(
                        ni, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0)))
                    ),
                    ntn.Assign(
                        nj, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1)))
                    ),
                    ntn.Unpack(A_,A),
                    ntn.Assign(
                        nj, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0)))
                    ),
                    ntn.Assign(
                        ni, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1)))
                    ),

                )
            )
        )








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