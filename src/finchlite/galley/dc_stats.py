import math
import operator
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

import finchlite.finch_notation as ntn
from finchlite.algebra import is_annihilator
from finchlite.compile import dimension
from finchlite.finch_notation.nodes import (
    Literal,
)

from .tensor_def import TensorDef
from .tensor_stats import TensorStats


@dataclass(frozen=True)
class DC:
    """
    A degree constraint (DC) record representing structural cardinality

    Attributes:
        from_indices: Conditioning index names.
        to_indices: Index names whose distinct combinations are counted
            when `from_indices` are fixed.
        value: Estimated number of distinct combinations for `to_indices`
            given the fixed `from_indices`.
    """

    from_indices: frozenset[str]
    to_indices: frozenset[str]
    value: float


class DCStats(TensorStats):
    """
    Structural statistics derived from a tensor using degree constraint (DCs).

    DCStats scans a tensor and computes degree constraint (DC) records that
    summarize how index sets relate. These DCs can be used to estimate the
    number of non-fill values without materializing sparse coordinates.
    """

    def __init__(self, tensor: Any, fields: Iterable[str]):
        """
        Initialize DCStats from a tensor and its axis names, build the TensorDef,
        and compute degree constraint (DC) records from the tensor’s structure.
        """
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        self.dcs = self._structure_to_dcs(tensor)

    @staticmethod
    def from_def(tensordef: TensorDef, dcs: set[DC]) -> "DCStats":
        """
        Build DCStats directly from a TensorDef and an existing DC set.
        """
        self = object.__new__(DCStats)
        self.tensordef = tensordef.copy()
        self.dcs = set(dcs)
        return self

    def _structure_to_dcs(self, arr: Any) -> set[DC]:
        """
        Dispatch DC extraction based on tensor dimensionality.

        Returns:
            set[DC]: One of the following, depending on `self.tensor.ndim`:
                • Empty set, if the tensor is empty (`self.tensor.size == 0`)
                • 1D → _vector_structure_to_dcs()
                • 2D → _matrix_structure_to_dcs()
                • 3D → _3d_structure_to_dcs()
                • 4D → _4d_structure_to_dcs()

        Raises:
            NotImplementedError: If dimensionality is not in {1, 2, 3, 4}.
        """
        if arr is None or getattr(arr, "size", 0) == 0:
            return set()

        ndim = getattr(arr, "ndim", None)
        if ndim == 1:
            return self._vector_structure_to_dcs(arr)
        if ndim == 2:
            return self._matrix_structure_to_dcs(arr)
        if ndim == 3:
            return self._3d_structure_to_dcs(arr)
        if ndim == 4:
            return self._4d_structure_to_dcs(arr)
        raise NotImplementedError(f"DC analysis not implemented for {ndim}D tensors")

    def _vector_structure_to_dcs(self, arr: Any) -> set[DC]:
        """
        Build and execute a Finch-notation program that analyzes the structural
        relationships within a one-dimensional tensor.

        Returns:
            set[DC]: The degree constraint (DC) records derived from the 1-D tensor.
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
        cnt = mod.vector_structure_to_dcs(arr)
        result = next(iter(self.tensordef.dim_sizes))

        return {DC(frozenset(), frozenset([result]), float(cnt))}

    def _matrix_structure_to_dcs(self, arr: Any) -> set[DC]:
        """
        Build and execute a Finch-notation program that analyzes the structural
        relationships within a two-dimensional tensor.

        Returns:
            set[DC]: The degree constraint (DC) records derived from the 2-D tensor.
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

        d_ij = mod.matrix_total_nnz(arr)
        d_i_, d_i_j_, d_j_, d_j_i_ = mod.matrix_structure_to_dcs(arr)
        i_result, j_result = tuple(self.tensordef.dim_sizes)

        return {
            DC(frozenset(), frozenset([i_result, j_result]), float(d_ij)),
            DC(frozenset(), frozenset([i_result]), float(d_i_)),
            DC(frozenset(), frozenset([j_result]), float(d_j_)),
            DC(frozenset([i_result]), frozenset([i_result, j_result]), float(d_i_j_)),
            DC(frozenset([j_result]), frozenset([i_result, j_result]), float(d_j_i_)),
        }

    def _3d_structure_to_dcs(self, arr: Any) -> set[DC]:
        """
        Build and execute a Finch-notation program that analyzes structural
        relationships within a three-dimensional tensor.

        Returns:
            set[DC]: The degree constraint (DC) records derived from the 3-D tensor.
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
                                                        ntn.Access(
                                                            A_, ntn.Read(), (k, j, i)
                                                        )
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
                                                                    A_,
                                                                    ntn.Read(),
                                                                    (k, j, i),
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
                                                                    A_,
                                                                    ntn.Read(),
                                                                    (k, j, i),
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
                                                                    A_,
                                                                    ntn.Read(),
                                                                    (k, j, i),
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
                                    ntn.Literal(
                                        lambda a, b, c, d, e, f: (a, b, c, d, e, f)
                                    ),
                                    (d_i, d_i_jk, d_j, d_j_ik, d_k, d_k_ij),
                                )
                            ),
                        )
                    ),
                ),
            )
        )
        mod = ntn.NotationInterpreter()(prgm)

        d_ijk = mod._3d_total_nnz(arr)
        d_i_, d_i_jk_, d_j_, d_j_ik_, d_k_, d_k_ij_ = mod._3d_structure_to_dcs(arr)
        i_result, j_result, k_result = tuple(self.tensordef.dim_sizes)
        return {
            DC(frozenset(), frozenset([i_result]), float(d_i_)),
            DC(frozenset(), frozenset([j_result]), float(d_j_)),
            DC(frozenset(), frozenset([k_result]), float(d_k_)),
            DC(frozenset([i_result]), frozenset([j_result, k_result]), float(d_i_jk_)),
            DC(frozenset([j_result]), frozenset([i_result, k_result]), float(d_j_ik_)),
            DC(frozenset([k_result]), frozenset([i_result, j_result]), float(d_k_ij_)),
            DC(frozenset([]), frozenset([i_result, j_result, k_result]), float(d_ijk)),
        }

    def _4d_structure_to_dcs(self, arr: Any) -> set[DC]:
        """
        Build and execute a Finch-notation program that analyzes structural
        relationships within a four-dimensional tensor.

        Returns:
            set[DC]: The degree constraint (DC) records derived from the 4-D tensor.
        """
        A = ntn.Variable("A", np.ndarray)
        A_ = ntn.Slot("A_", np.ndarray)

        i = ntn.Variable("i", np.int64)
        j = ntn.Variable("j", np.int64)
        k = ntn.Variable("k", np.int64)
        w = ntn.Variable("w", np.int64)

        ni = ntn.Variable("ni", np.int64)
        nj = ntn.Variable("nj", np.int64)
        nk = ntn.Variable("nk", np.int64)
        nw = ntn.Variable("nw", np.int64)

        dijkw = ntn.Variable("dijkw", np.int64)

        xi = ntn.Variable("xi", np.int64)
        yj = ntn.Variable("yj", np.int64)
        zk = ntn.Variable("zk", np.int64)
        uw = ntn.Variable("uw", np.int64)

        d_i = ntn.Variable("d_i", np.int64)
        d_i_jkw = ntn.Variable("d_i_jkw", np.int64)
        d_j = ntn.Variable("d_j", np.int64)
        d_j_ikw = ntn.Variable("d_j_ikw", np.int64)
        d_k = ntn.Variable("d_k", np.int64)
        d_k_ijw = ntn.Variable("d_k_ijw", np.int64)
        d_w = ntn.Variable("d_w", np.int64)
        d_w_ijk = ntn.Variable("d_l_ijw", np.int64)

        prgm = ntn.Module(
            (
                ntn.Function(
                    ntn.Variable("_4d_total_nnz", np.int64),
                    (A,),
                    ntn.Block(
                        (
                            ntn.Assign(
                                nw,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0))),
                            ),
                            ntn.Assign(
                                nk,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1))),
                            ),
                            ntn.Assign(
                                nj,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(2))),
                            ),
                            ntn.Assign(
                                ni,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(3))),
                            ),
                            ntn.Assign(dijkw, ntn.Literal(np.int64(0))),
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
                                        ntn.Loop(
                                            w,
                                            nw,
                                            ntn.Assign(
                                                dijkw,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (
                                                        dijkw,
                                                        ntn.Unwrap(
                                                            ntn.Access(
                                                                A_,
                                                                ntn.Read(),
                                                                (w, k, j, i),
                                                            )
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            ntn.Repack(A_, A),
                            ntn.Return(dijkw),
                        )
                    ),
                ),
                ntn.Function(
                    ntn.Variable("_4d_structure_to_dcs", tuple),
                    (A,),
                    ntn.Block(
                        (
                            ntn.Assign(
                                nw,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0))),
                            ),
                            ntn.Assign(
                                nk,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1))),
                            ),
                            ntn.Assign(
                                nj,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(2))),
                            ),
                            ntn.Assign(
                                ni,
                                ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(3))),
                            ),
                            ntn.Unpack(A_, A),
                            ntn.Assign(d_i, ntn.Literal(np.int64(0))),
                            ntn.Assign(d_i_jkw, ntn.Literal(np.int64(0))),
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
                                                ntn.Loop(
                                                    w,
                                                    nw,
                                                    ntn.Assign(
                                                        xi,
                                                        ntn.Call(
                                                            ntn.Literal(operator.add),
                                                            (
                                                                xi,
                                                                ntn.Unwrap(
                                                                    ntn.Access(
                                                                        A_,
                                                                        ntn.Read(),
                                                                        (w, k, j, i),
                                                                    )
                                                                ),
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
                                            d_i_jkw,
                                            ntn.Call(ntn.Literal(max), (d_i_jkw, xi)),
                                        ),
                                    )
                                ),
                            ),
                            ntn.Assign(d_j, ntn.Literal(np.int64(0))),
                            ntn.Assign(d_j_ikw, ntn.Literal(np.int64(0))),
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
                                                ntn.Loop(
                                                    w,
                                                    nw,
                                                    ntn.Assign(
                                                        yj,
                                                        ntn.Call(
                                                            ntn.Literal(operator.add),
                                                            (
                                                                yj,
                                                                ntn.Unwrap(
                                                                    ntn.Access(
                                                                        A_,
                                                                        ntn.Read(),
                                                                        (w, k, j, i),
                                                                    )
                                                                ),
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
                                            d_j_ikw,
                                            ntn.Call(ntn.Literal(max), (d_j_ikw, yj)),
                                        ),
                                    )
                                ),
                            ),
                            ntn.Assign(d_k, ntn.Literal(np.int64(0))),
                            ntn.Assign(d_k_ijw, ntn.Literal(np.int64(0))),
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
                                                ntn.Loop(
                                                    w,
                                                    nw,
                                                    ntn.Assign(
                                                        zk,
                                                        ntn.Call(
                                                            ntn.Literal(operator.add),
                                                            (
                                                                zk,
                                                                ntn.Unwrap(
                                                                    ntn.Access(
                                                                        A_,
                                                                        ntn.Read(),
                                                                        (w, k, j, i),
                                                                    )
                                                                ),
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
                                            d_k_ijw,
                                            ntn.Call(ntn.Literal(max), (d_k_ijw, zk)),
                                        ),
                                    )
                                ),
                            ),
                            ntn.Assign(d_w, ntn.Literal(np.int64(0))),
                            ntn.Assign(d_w_ijk, ntn.Literal(np.int64(0))),
                            ntn.Loop(
                                w,
                                nw,
                                ntn.Block(
                                    (
                                        ntn.Assign(uw, ntn.Literal(np.int64(0))),
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
                                                        uw,
                                                        ntn.Call(
                                                            ntn.Literal(operator.add),
                                                            (
                                                                uw,
                                                                ntn.Unwrap(
                                                                    ntn.Access(
                                                                        A_,
                                                                        ntn.Read(),
                                                                        (w, k, j, i),
                                                                    )
                                                                ),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        ntn.If(
                                            ntn.Call(
                                                ntn.Literal(operator.ne),
                                                (uw, ntn.Literal(np.int64(0))),
                                            ),
                                            ntn.Assign(
                                                d_w,
                                                ntn.Call(
                                                    ntn.Literal(operator.add),
                                                    (d_w, ntn.Literal(np.int64(1))),
                                                ),
                                            ),
                                        ),
                                        ntn.Assign(
                                            d_w_ijk,
                                            ntn.Call(ntn.Literal(max), (d_w_ijk, uw)),
                                        ),
                                    )
                                ),
                            ),
                            ntn.Repack(A_, A),
                            ntn.Return(
                                ntn.Call(
                                    ntn.Literal(
                                        lambda a, b, c, d, e, f, g, h: (
                                            a,
                                            b,
                                            c,
                                            d,
                                            e,
                                            f,
                                            g,
                                            h,
                                        )
                                    ),
                                    (
                                        d_i,
                                        d_i_jkw,
                                        d_j,
                                        d_j_ikw,
                                        d_k,
                                        d_k_ijw,
                                        d_w,
                                        d_w_ijk,
                                    ),
                                )
                            ),
                        )
                    ),
                ),
            )
        )
        mod = ntn.NotationInterpreter()(prgm)

        d_ijkw = mod._4d_total_nnz(arr)
        d_i_, d_i_jkw_, d_j_, d_j_ikw_, d_k_, d_k_ijw_, d_w_, d_w_ijk_ = (
            mod._4d_structure_to_dcs(arr)
        )

        i_result, j_result, k_result, w_result = tuple(self.tensordef.dim_sizes)
        return {
            DC(frozenset(), frozenset([i_result]), float(d_i_)),
            DC(frozenset(), frozenset([j_result]), float(d_j_)),
            DC(frozenset(), frozenset([k_result]), float(d_k_)),
            DC(frozenset(), frozenset([w_result]), float(d_w_)),
            DC(
                frozenset([i_result]),
                frozenset([j_result, k_result, w_result]),
                float(d_i_jkw_),
            ),
            DC(
                frozenset([j_result]),
                frozenset([i_result, k_result, w_result]),
                float(d_j_ikw_),
            ),
            DC(
                frozenset([k_result]),
                frozenset([i_result, j_result, w_result]),
                float(d_k_ijw_),
            ),
            DC(
                frozenset([w_result]),
                frozenset([i_result, j_result, k_result]),
                float(d_w_ijk_),
            ),
            DC(
                frozenset([]),
                frozenset([i_result, j_result, k_result, w_result]),
                float(d_ijkw),
            ),
        }

    @staticmethod
    def _merge_dc_join(new_def: "TensorDef", all_stats: list["DCStats"]) -> "DCStats":
        """
        Merge DCs for join-like operators

        Args:
            new_def: The merged TensorDef produced by TensorDef.mapjoin(...).
            all_stats: DCStats inputs whose DC sets are to be merged.

        Returns:
            A new DCStats built on `new_def` with DCs:
            For every key (X, Y) that appears in at least one input, set
            d_out(X, Y) to the smallest of the values provided for that key
            by the inputs that define it.
        """
        if len(all_stats) == 1:
            return DCStats.from_def(new_def, set(all_stats[0].dcs))

        new_dc: dict[tuple[frozenset[str], frozenset[str]], float] = {}
        for stats in all_stats:
            for dc in stats.dcs:
                dc_key = (dc.from_indices, dc.to_indices)
                current_dc = new_dc.get(dc_key, math.inf)
                if dc.value < current_dc:
                    new_dc[dc_key] = dc.value

        new_stats = {DC(X, Y, d) for (X, Y), d in new_dc.items()}
        return DCStats.from_def(new_def, new_stats)

    @staticmethod
    def _merge_dc_union(new_def: "TensorDef", all_stats: list["DCStats"]) -> "DCStats":
        """
        Merge DCs for union-like operators.

        Args:
            new_def: The output TensorDef produced by TensorDef.mapjoin(...).
            all_stats: The DCStats inputs to merge.

        Returns:
            A new DCStats built on `new_def` whose DC set reflects union semantics.
            If there is only one input, this returns a copy of its DCs attached to
            `new_def`.
        """
        if len(all_stats) == 1:
            return DCStats.from_def(new_def, set(all_stats[0].dcs))

        dc_keys: Counter[tuple[frozenset[str], frozenset[str]]] = Counter()
        stats_dcs: list[dict[tuple[frozenset[str], frozenset[str]], float]] = []
        for stats in all_stats:
            dcs: dict[tuple[frozenset[str], frozenset[str]], float] = {}
            Z = new_def.index_set - stats.tensordef.index_set
            Z_dim_size = new_def.get_dim_space_size(Z)
            for dc in stats.dcs:
                new_key = (dc.from_indices, dc.to_indices)
                dcs[new_key] = dc.value
                dc_keys[new_key] += 1

                ext_dc_key = (dc.from_indices, dc.to_indices | frozenset(Z))
                if ext_dc_key not in dcs:
                    dc_keys[ext_dc_key] += 1
                prev = dcs.get(ext_dc_key, math.inf)
                dcs[ext_dc_key] = min(prev, dc.value * Z_dim_size)
            stats_dcs.append(dcs)

        new_dcs: dict[tuple[frozenset[str], frozenset[str]], float] = {}
        for key, count in dc_keys.items():
            if count == len(all_stats):
                total = sum(d.get(key, 0.0) for d in stats_dcs)
                X, Y = key
                if Y.issubset(new_def.index_set):
                    total = min(total, new_def.get_dim_space_size(Y))
                new_dcs[key] = min(float(2**64), total)

        new_stats = {DC(X, Y, d) for (X, Y), d in new_dcs.items()}
        return DCStats.from_def(new_def, new_stats)

    @staticmethod
    def mapjoin(op: Callable[..., Any], *all_stats: "TensorStats") -> "TensorStats":
        """
        Merge DC statistics for an elementwise binary operation.

        Args:
            op: The elementwise operator (e.g., operator.add, operator.mul).
            all_stats: Input statistics objects to be merged. Must be DCStats at runtime

        Returns:
            A DCStats instance whose TensorDef is the union of input dims and whose DC
            set reflects the operator semantics:
            - If every informative argument is *join-like* (its fill is an annihilator
                for `op`), merge with join rules (take minima over matching DC keys).
            - If every informative argument is *union-like* (fill not an annihilator),
                merge with union rules (infer/extend DCs to missing dims, sum compatible
                DCs, clamp by dense capacity).
            - If mixed, and the join-like arguments cover all output indices, prefer
                join merge; otherwise perform union merge over all arguments.
        """
        new_def = TensorDef.mapjoin(op, *(s.tensordef for s in all_stats))
        join_like_args: list[TensorStats] = []
        union_like_args: list[TensorStats] = []
        for stats in all_stats:
            if len(stats.tensordef.index_set) == 0:
                continue
            if is_annihilator(op, stats.tensordef.fill_value):
                join_like_args.append(stats)
            else:
                union_like_args.append(stats)
        join_like_dc: list[DCStats] = cast(list["DCStats"], join_like_args)
        union_like_dc: list[DCStats] = cast(list["DCStats"], union_like_args)

        if len(union_like_args) == 0 and len(join_like_args) == 0:
            return DCStats.from_def(new_def, set())
        if len(union_like_args) == 0:
            return DCStats._merge_dc_join(new_def, join_like_dc)
        if len(join_like_args) == 0:
            return DCStats._merge_dc_union(new_def, union_like_dc)
        join_cover = set().union(*(s.tensordef.index_set for s in join_like_dc))
        if join_cover == new_def.index_set:
            return DCStats._merge_dc_join(new_def, join_like_dc)
        return DCStats._merge_dc_union(new_def, join_like_dc + union_like_dc)

    @staticmethod
    def aggregate(op, dims, *args, **kwargs):
        pass

    @staticmethod
    def issimilar(*args, **kwargs):
        pass

    def estimate_non_fill_values(self) -> float:
        """
        Estimate the number of non-fill values using DCs.

        This uses the stored degree constraint (DC) as multiplicative factors to
        grow coverage over the target indices and finds the smallest product that
        covers all target indices. The result is clamped by the tensor’s dense
        capacity (the product of the target dimensions).

        Returns:
            the estimated number of non-fill entries in the tensor.
        """
        idx: frozenset[str] = frozenset(self.tensordef.dim_sizes.keys())
        if len(idx) == 0:
            return 0.0

        best: dict[frozenset[str], float] = {frozenset(): 1.0}
        frontier: set[frozenset[str]] = {frozenset()}

        while True:
            current_bound = best.get(idx, math.inf)
            new_frontier: set[frozenset[str]] = set()

            for node in frontier:
                for dc in self.dcs:
                    if node.issuperset(dc.from_indices):
                        y = node.union(dc.to_indices)
                        if best[node] > float(2 ** (64 - 2)) or float(dc.value) > float(
                            2 ** (64 - 2)
                        ):
                            y_weight = float(2**64)
                        else:
                            y_weight = best[node] * dc.value
                        if min(current_bound, best.get(y, math.inf)) > y_weight:
                            best[y] = y_weight
                            new_frontier.add(y)
            if len(new_frontier) == 0:
                break
            frontier = new_frontier

        min_weight = float(self.get_dim_space_size(idx))
        for node, weight in best.items():
            if node.issuperset(idx):
                min_weight = min(min_weight, weight)
        return min_weight
