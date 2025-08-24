import operator
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

import finch.finch_notation as ntn
from finch.compile import dimension
from finch.finch_notation.nodes import (
    Literal,
)

from .tensor_def import TensorDef
from .tensor_stats import TensorStats


@dataclass(frozen=True)
class DC:
    from_indices: frozenset[str]
    to_indices: frozenset[str]
    value: float


class DCStats(TensorStats):
    def __init__(self, tensor: Any, fields: Iterable[str]):
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        self.fields = list(fields)
        self.tensor = np.asarray(tensor)
        self.dcs = self._structure_to_dcs()

    @classmethod
    def from_tensor(cls, tensor: Any, fields: Iterable[str]) -> None:
        return None

    def _structure_to_dcs(self) -> set[DC]:
        if self.tensor.size == 0:
            return set()

        ndim = self.tensor.ndim
        if ndim == 1:
            return self._vector_structure_to_dcs()
        if ndim == 2:
            return self._matrix_structure_to_dcs()
        if ndim == 3:
            return self._3d_structure_to_dcs()
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
                                            ntn.Unwrap(
                                                ntn.Access(A_, ntn.Read(), (i,))
                                            ),
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
        """
        Build a Finch Notation program that extracts structural dependencies
        from a 2-D tensor, execute it, and return a set of DC.
        """

        A = ntn.Variable("A", np.ndarray)
        A_ = ntn.Slot("A_", np.ndarray)

        i = ntn.Variable("i", np.int64)
        j = ntn.Variable("j", np.int64)
        ni = ntn.Variable("ni", np.int64)
        nj = ntn.Variable("nj", np.int64)

        dij = ntn.Variable("dij", np.int64)

        xi = ntn.Variable("xi", np.int64)
        yj = ntn.Variable("yj", np.int64)

        d_i = ntn.Variable("d_i", np.int64)
        d_i_j = ntn.Variable("d_i_j", np.int64)
        d_j = ntn.Variable("d_j", np.int64)
        d_j_i = ntn.Variable("d_j_i", np.int64)

        prgm = ntn.Module(
            (
                ntn.Function(
                    ntn.Variable("matrix_total_nnz", np.int64),
                    (A,),
                    ntn.Block(
                        (
                            ntn.Assign(
                                nj,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0))),
                            ),
                            ntn.Assign(
                                ni,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1))),
                            ),
                            ntn.Assign(dij, ntn.Literal(np.int64(0))),
                            ntn.Unpack(A_, A),
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
                                                    ntn.Access(A_, ntn.Read(), (j, i))
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            ntn.Repack(A_, A),
                            ntn.Return(dij),
                        )
                    ),
                ),
                ntn.Function(
                    ntn.Variable("matrix_structure_to_dcs", tuple),
                    (A,),
                    ntn.Block(
                        (
                            ntn.Assign(
                                nj,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0))),
                            ),
                            ntn.Assign(
                                ni,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1))),
                            ),
                            ntn.Unpack(A_, A),
                            ntn.Assign(d_i, ntn.Literal(np.int64(0))),
                            ntn.Assign(d_i_j, ntn.Literal(np.int64(0))),
                            ntn.Loop(
                                i,
                                ni,
                                ntn.Block(
                                    (
                                        ntn.Assign(xi, ntn.Literal(np.int64(0))),
                                        ntn.Loop(
                                            j,
                                            nj,
                                            ntn.Assign(
                                                xi,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (
                                                        xi,
                                                        ntn.Unwrap(
                                                            ntn.Access(
                                                                A_, ntn.Read(), (j, i)
                                                            )
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        ntn.If(
                                            ntn.Call(
                                                ntn.Literal(operator.ne),
                                                (xi, ntn.Literal(np.int64(0))),
                                            ),
                                            ntn.Assign(
                                                d_i,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (d_i, ntn.Literal(np.int64(1))),
                                                ),
                                            ),
                                        ),
                                        ntn.Assign(
                                            d_i_j,
                                            ntn.Call(ntn.Literal(max), (d_i_j, xi)),
                                        ),
                                    )
                                ),
                            ),
                            ntn.Assign(d_j, ntn.Literal(np.int64(0))),
                            ntn.Assign(d_j_i, ntn.Literal(np.int64(0))),
                            ntn.Loop(
                                j,
                                nj,
                                ntn.Block(
                                    (
                                        ntn.Assign(yj, ntn.Literal(np.int64(0))),
                                        ntn.Loop(
                                            i,
                                            ni,
                                            ntn.Assign(
                                                yj,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (
                                                        yj,
                                                        ntn.Unwrap(
                                                            ntn.Access(
                                                                A_, ntn.Read(), (j, i)
                                                            )
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        ntn.If(
                                            ntn.Call(
                                                ntn.Literal(operator.ne),
                                                (yj, ntn.Literal(np.int64(0))),
                                            ),
                                            ntn.Assign(
                                                d_j,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (d_j, ntn.Literal(np.int64(1))),
                                                ),
                                            ),
                                        ),
                                        ntn.Assign(
                                            d_j_i,
                                            ntn.Call(ntn.Literal(max), (d_j_i, yj)),
                                        ),
                                    )
                                ),
                            ),
                            ntn.Repack(A_, A),
                            ntn.Return(
                                ntn.Call(
                                    ntn.Literal(lambda a, b, c, d: (a, b, c, d)),
                                    (d_i, d_i_j, d_j, d_j_i),
                                )
                            ),
                        )
                    ),
                ),
            )
        )
        mod = ntn.NotationInterpreter()(prgm)

        d_ij = mod.matrix_total_nnz(self.tensor)
        d_i_, d_i_j_, d_j_, d_j_i_ = mod.matrix_structure_to_dcs(self.tensor)
        i_result, j_result = self.fields
        return {
            DC(frozenset(), frozenset([i_result, j_result]), float(d_ij)),
            DC(frozenset(), frozenset([i_result]), float(d_i_)),
            DC(frozenset(), frozenset([j_result]), float(d_j_)),
            DC(frozenset([i_result]), frozenset([i_result, j_result]), float(d_i_j_)),
            DC(frozenset([j_result]), frozenset([i_result, j_result]), float(d_j_i_)),
        }

    def _3d_structure_to_dcs(self) -> set[DC]:
        """
        Build a Finch Notation program that extracts structural dependencies
        from a 3-D tensor, execute it, and return a set of DC.
        """
        A = ntn.Variable("A", np.ndarray)
        A_ = ntn.Slot("A_", np.ndarray)

        i = ntn.Variable("i", np.int64)
        j = ntn.Variable("j", np.int64)
        k = ntn.Variable("k", np.int64)

        ni = ntn.Variable("ni", np.int64)
        nj = ntn.Variable("nj", np.int64)
        nk = ntn.Variable("nk", np.int64)

        dijk = ntn.Variable("dijk", np.int64)

        xi = ntn.Variable("xi", np.int64)
        yj = ntn.Variable("yj", np.int64)
        zk = ntn.Variable("zk", np.int64)

        d_i = ntn.Variable("d_i", np.int64)
        d_i_jk = ntn.Variable("d_i_jk", np.int64)
        d_j = ntn.Variable("d_j", np.int64)
        d_j_ik = ntn.Variable("d_j_ik", np.int64)
        d_k = ntn.Variable("d_k", np.int64)
        d_k_ij = ntn.Variable("d_k_ij", np.int64)

        prgm = ntn.Module(
            (
                ntn.Function(
                    ntn.Variable("_3d_total_nnz", np.int64),
                    (A,),
                    ntn.Block(
                        (
                            ntn.Assign(
                                nk,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0))),
                            ),
                            ntn.Assign(
                                nj,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1))),
                            ),
                            ntn.Assign(
                                ni,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(2))),
                            ),
                            ntn.Assign(dijk, ntn.Literal(np.int64(0))),
                            ntn.Unpack(A_, A),
                            ntn.Loop(
                                i,
                                ni,
                                ntn.Loop(
                                    j,
                                    nj,
                                    ntn.Loop(
                                        k,
                                        nk,
                                        ntn.Assign(
                                            dijk,
                                            ntn.Call(
                                                ntn.Literal(operator.add),
                                                (
                                                    dijk,
                                                    ntn.Unwrap(
                                                        ntn.Access(A_, ntn.Read(), (k, j, i))
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            ntn.Repack(A_, A),
                            ntn.Return(dijk),
                        )
                    ),
                ),
                ntn.Function(
                    ntn.Variable("_3d_structure_to_dcs", tuple),
                    (A,),
                    ntn.Block(
                        (
                            ntn.Assign(
                                nk,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0))),
                            ),
                            ntn.Assign(
                                nj,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1))),
                            ),
                            ntn.Assign(
                                ni,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(2))),
                            ),
                            ntn.Unpack(A_, A),
                            ntn.Assign(d_i, ntn.Literal(np.int64(0))),
                            ntn.Assign(d_i_jk, ntn.Literal(np.int64(0))),
                            ntn.Loop(
                                i,
                                ni,
                                ntn.Block(
                                    (
                                        ntn.Assign(xi, ntn.Literal(np.int64(0))),
                                        ntn.Loop(
                                            j,
                                            nj,
                                            ntn.Loop(
                                                k,
                                                nk,
                                                ntn.Assign(
                                                    xi,
                                                    ntn.Call(
                                                        ntn.Literal(operator.add),
                                                        (
                                                            xi,
                                                            ntn.Unwrap(
                                                                ntn.Access(
                                                                    A_, ntn.Read(), (k, j, i)
                                                                )
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        ntn.If(
                                            ntn.Call(
                                                ntn.Literal(operator.ne),
                                                (xi, ntn.Literal(np.int64(0))),
                                            ),
                                            ntn.Assign(
                                                d_i,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (d_i, ntn.Literal(np.int64(1))),
                                                ),
                                            ),
                                        ),
                                        ntn.Assign(
                                            d_i_jk,
                                            ntn.Call(ntn.Literal(max), (d_i_jk, xi)),
                                        ),
                                    )
                                ),
                            ),
                            ntn.Assign(d_j, ntn.Literal(np.int64(0))),
                            ntn.Assign(d_j_ik, ntn.Literal(np.int64(0))),
                            ntn.Loop(
                                j,
                                nj,
                                ntn.Block(
                                    (
                                        ntn.Assign(yj, ntn.Literal(np.int64(0))),
                                        ntn.Loop(
                                            i,
                                            ni,
                                            ntn.Loop(
                                                k,
                                                nk,
                                                ntn.Assign(
                                                    yj,
                                                    ntn.Call(
                                                        ntn.Literal(operator.add),
                                                        (
                                                            yj,
                                                            ntn.Unwrap(
                                                                ntn.Access(
                                                                    A_, ntn.Read(), (k, j, i)
                                                                )
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        ntn.If(
                                            ntn.Call(
                                                ntn.Literal(operator.ne),
                                                (yj, ntn.Literal(np.int64(0))),
                                            ),
                                            ntn.Assign(
                                                d_j,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (d_j, ntn.Literal(np.int64(1))),
                                                ),
                                            ),
                                        ),
                                        ntn.Assign(
                                            d_j_ik,
                                            ntn.Call(ntn.Literal(max), (d_j_ik, yj)),
                                        ),
                                    )
                                ),
                            ),
                            ntn.Assign(d_k, ntn.Literal(np.int64(0))),
                            ntn.Assign(d_k_ij, ntn.Literal(np.int64(0))),
                            ntn.Loop(
                                k,
                                nk,
                                ntn.Block(
                                    (
                                        ntn.Assign(zk, ntn.Literal(np.int64(0))),
                                        ntn.Loop(
                                            i,
                                            ni,
                                            ntn.Loop(
                                                j,
                                                nj,
                                                ntn.Assign(
                                                    zk,
                                                    ntn.Call(
                                                        ntn.Literal(operator.add),
                                                        (
                                                            zk,
                                                            ntn.Unwrap(
                                                                ntn.Access(
                                                                    A_, ntn.Read(), (k, j, i)
                                                                )
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        ntn.If(
                                            ntn.Call(
                                                ntn.Literal(operator.ne),
                                                (zk, ntn.Literal(np.int64(0))),
                                            ),
                                            ntn.Assign(
                                                d_k,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (d_k, ntn.Literal(np.int64(1))),
                                                ),
                                            ),
                                        ),
                                        ntn.Assign(
                                            d_k_ij,
                                            ntn.Call(ntn.Literal(max), (d_k_ij, zk)),
                                        ),
                                    )
                                ),
                            ),
                            ntn.Repack(A_, A),
                            ntn.Return(
                                ntn.Call(
                                    ntn.Literal(lambda a, b, c, d, e, f: (a, b, c, d, e, f)),
                                    (d_i, d_i_jk, d_j, d_j_ik, d_k, d_k_ij),
                                )
                            )
                        )
                    ),
                ),
            )
        )
        mod = ntn.NotationInterpreter()(prgm)

        d_ijk = mod._3d_total_nnz(self.tensor)
        d_i_, d_i_jk_, d_j_, d_j_ik_, d_k_, d_k_ij_ = mod._3d_structure_to_dcs(self.tensor)
        i_result, j_result, k_result = self.fields
        return {
            DC(frozenset(), frozenset([i_result]), float(d_i_)),
            DC(frozenset(), frozenset([j_result]), float(d_j_)),
            DC(frozenset(), frozenset([k_result]), float(d_k_)),
            DC(frozenset([i_result]), frozenset([j_result, k_result]), float(d_i_jk_)),
            DC(frozenset([j_result]), frozenset([i_result, k_result]), float(d_j_ik_)),
            DC(frozenset([k_result]), frozenset([i_result, j_result]), float(d_k_ij_)),
            DC(frozenset([]), frozenset([i_result, j_result, k_result]), float(d_ijk)),
        }

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
        """
        Return:
            the estimated number of non-fill values using DCs.
        """
        pass

