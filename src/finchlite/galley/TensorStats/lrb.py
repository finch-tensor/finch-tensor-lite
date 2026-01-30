import operator
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ... import finch_notation as ntn
from ...compile import BufferizedNDArray, dimension
from ...interface import asarray
from .dc_stats import DC, DCStats
from .tensor_def import TensorDef


@dataclass(frozen=True)
class RegionNNZ:
    """
    Per-region data of nonzero counts along one axis.
    """

    axis: str
    regions: int
    width: int
    nonempty_counts: np.ndarray
    max_region_nnz: np.ndarray
    sum_region_nnz: np.ndarray


def compute_axis_region_nnz(
    arr: Any,
    tensordef: TensorDef,
    fields: Iterable[str],
    axis: str,
) -> np.ndarray:
    """
    Compute region nnz for a single axis using Finch notation.

    Returns:
        region_nnz: np.ndarray shape (|axis|,), where
            region_nnz[u] = # of non-zero entries in arr whose coordinate on `axis` is u
    """
    fields = list(fields)
    ndims = len(fields)
    if axis not in fields:
        raise ValueError(f"axis '{axis}' not found in fields={fields}")
    axis_pos = fields.index(axis)

    dim_loop_variables = [ntn.Variable(fields[i], np.int64) for i in range(ndims)]
    dim_size_variables = [
        ntn.Variable(f"n_{fields[i]}", np.int64) for i in range(ndims)
    ]

    A = ntn.Variable("A", arr.ftype)
    A_ = ntn.Slot("A_", arr.ftype)

    region_nnz = ntn.Variable("region_nnz", BufferizedNDArray)
    region_nnz_ = ntn.Slot("region_nnz_", BufferizedNDArray)
    A_access = ntn.Unwrap(ntn.Access(A_, ntn.Read(), tuple(dim_loop_variables)))
    n_axis = tensordef.dim_sizes[axis]
    stmts = [
        ntn.Assign(
            dim_size_variables[i],
            ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(i))),
        )
        for i in range(ndims)
    ]

    stmts.extend(
        (
            ntn.Unpack(A_, A),
            ntn.Unpack(region_nnz_, region_nnz),
            ntn.Declare(
                region_nnz_,
                ntn.Literal(np.int64(0)),
                ntn.Literal(operator.add),
                (ntn.Literal(n_axis),),
            ),
        )
    )

    inc_expr = ntn.Increment(
        ntn.Access(
            region_nnz_,
            ntn.Update(ntn.Literal(operator.add)),
            (dim_loop_variables[axis_pos],),
        ),
        ntn.Call(
            ntn.Literal(operator.ne),
            (A_access, ntn.Literal(tensordef.fill_value)),
        ),
    )

    body: ntn.NotationStatement = inc_expr
    for i in range(ndims):
        body = ntn.Loop(dim_loop_variables[i], dim_size_variables[i], body)

    stmts.append(body)
    stmts.extend(
        [
            ntn.Freeze(region_nnz_, ntn.Literal(operator.add)),
            ntn.Repack(region_nnz_, region_nnz),
            ntn.Repack(A_, A),
            ntn.Return(region_nnz),
        ]
    )

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("axis_region_nnz", BufferizedNDArray),
                (A, region_nnz),
                ntn.Block(tuple(stmts)),
            ),
        )
    )

    mod = ntn.NotationInterpreter()(prgm)
    out_buf = asarray(np.zeros(n_axis, dtype=np.int64))
    out = mod.axis_region_nnz(arr, out_buf)

    return np.asarray(out, dtype=np.int64)


def nnz_to_regions(axis: str, nnz: np.ndarray, regions: int) -> RegionNNZ:
    """
    Partition axis values [0..N-1] into contiguous regions.

    width = ceil(N / regions)
    region_id = idx // width

    For each region, compute:
      - nonempty_counts[r] : number of regions with nnz > 0
      - max_region_nnz[r]   : maximum nnz in that region
    """
    nnz = np.asarray(nnz, dtype=np.int64)
    if regions <= 0:
        raise ValueError("regions must be positive")

    width = (nnz.shape[0] + regions - 1) // regions

    nonempty = np.zeros(regions, dtype=np.int64)
    max_regions = np.zeros(regions, dtype=np.int64)
    sum_regions = np.zeros(regions, dtype=np.int64)

    for idx in range(nnz.shape[0]):
        rid = idx // width
        sum_regions[rid] += nnz[idx]

        if nnz[idx] <= 0:
            continue

        nonempty[rid] += 1
        if nnz[idx] > max_regions[rid]:
            max_regions[rid] = nnz[idx]

    return RegionNNZ(
        axis=axis,
        regions=regions,
        width=width,
        nonempty_counts=nonempty,
        max_region_nnz=max_regions,
        sum_region_nnz=sum_regions,
    )


def lrb_get_stats(A: Any, fields: Iterable[str]) -> DCStats:
    return DCStats(A, fields)


def lrb_get_nnz(stats: DCStats) -> float:
    return stats.estimate_non_fill_values()


def lrb_matmul_stats(
    A_arr: Any,
    stats_A: DCStats,
    A_fields: list[str],
    B_arr: Any,
    stats_B: DCStats,
    B_fields: list[str],
    *,
    reduction_axis: str,
    regions: int,
) -> DCStats:
    """
    Localized Region Bound for AB = A[i,j] @ B[j,k] -> AB[i,k].

    Bound:
      nnz(AB) <= sum_{region r} ( |J_r|_nonempty * maxA_r * maxB_r )
    where:
      |J_r|_nonempty = number of j values in region r whose nnz > 0
      maxA_r = max over j in region r of nnz(A[:,j])
      maxB_r = max over j in region r of nnz(B[j,:])
    """
    if regions <= 0:
        raise ValueError("regions must be positive")

    j = reduction_axis
    free_A = [ax for ax in A_fields if ax != j]
    free_B = [ax for ax in B_fields if ax != j]
    i = free_A[0]
    k = free_B[0]
    i_size = stats_A.tensordef.dim_sizes[i]
    k_size = stats_B.tensordef.dim_sizes[k]

    A_j = compute_axis_region_nnz(A_arr, stats_A.tensordef, A_fields, axis=j)
    B_j = compute_axis_region_nnz(B_arr, stats_B.tensordef, B_fields, axis=j)
    regA = nnz_to_regions(j, A_j, regions)
    regB = nnz_to_regions(j, B_j, regions)

    total = 0.0
    for r in range(regions):
        amax = regA.max_region_nnz[r]
        bmax = regB.max_region_nnz[r]
        nnzA_r = regA.sum_region_nnz[r]
        nnzB_r = regB.sum_region_nnz[r]

        if amax == 0 or bmax == 0 or nnzA_r == 0 or nnzB_r == 0:
            continue

        total += min(nnzA_r * bmax, amax * nnzB_r)

    total = min(total, i_size * k_size)

    out_def = TensorDef(
        index_set=[i, k],
        dim_sizes={i: i_size, k: k_size},
        fill_value=stats_A.tensordef.fill_value,
    )

    out_dcs: set[DC] = {
        DC(frozenset(), frozenset([i, k]), float(total)),
        DC(frozenset(), frozenset([i]), float(i_size)),
        DC(frozenset(), frozenset([k]), float(k_size)),
    }

    return DCStats.from_def(out_def, out_dcs)
