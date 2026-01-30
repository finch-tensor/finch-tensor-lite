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
    Localized Region Bound for AB = A[i,j] @ B[j,k] → AB[i,k].

    The reduction axis j is partitioned into contiguous regions. For each region r,
    we derive an upper bound on the contribution of that region to nnz(AB) using
    coarse, worst-case sparsity summaries of A and B restricted to that region.

    Bound:
        nnz(AB) ≤ ∑_r min(
            nnz_A[r] · maxB_r,
            maxA_r · nnz_B[r]
        )

    where:
        nnz_A[r] = ∑_{j∈r} nnz(A[:, j])        (total nonzeros of A in region r)
        nnz_B[r] = ∑_{j∈r} nnz(B[j, :])        (total nonzeros of B in region r)
        maxA_r   = max_{j∈r} nnz(A[:, j])      (densest column of A in region r)
        maxB_r   = max_{j∈r} nnz(B[j, :])      (densest row of B in region r)

    The per-region bounds are summed and finally clamped by the dense output size
    |I|·|K| to ensure soundness.
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


def lrb_3d_matmul_stats(
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
    Localized Region Bound for 3D matmul:
        C[b,i,k] = sum_j A[b,i,j] * B[b,j,k].
    """
    if regions <= 0:
        raise ValueError("regions must be positive")

    j = reduction_axis

    out_axes = []
    seen = set()
    for ax in A_fields + B_fields:
        if ax != j and ax not in seen:
            seen.add(ax)
            out_axes.append(ax)

    # Sizes for clamping/output TensorDef
    out_dim_sizes: dict[str, int] = {}
    for ax in out_axes:
        if ax in stats_A.tensordef.dim_sizes:
            out_dim_sizes[ax] = stats_A.tensordef.dim_sizes[ax]
        elif ax in stats_B.tensordef.dim_sizes:
            out_dim_sizes[ax] = stats_B.tensordef.dim_sizes[ax]
        else:
            raise ValueError(f"Axis {ax} not found in A or B dim_sizes")

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

    total = min(
        total,
        out_dim_sizes[out_axes[0]]
        * out_dim_sizes[out_axes[1]]
        * out_dim_sizes[out_axes[2]],
    )

    out_def = TensorDef(
        index_set=out_axes,
        dim_sizes=out_dim_sizes,
        fill_value=stats_A.tensordef.fill_value,
    )

    out_dcs: set[DC] = {DC(frozenset(), frozenset(out_axes), float(total))}
    for ax in out_axes:
        out_dcs.add(DC(frozenset(), frozenset([ax]), float(out_dim_sizes[ax])))

    return DCStats.from_def(out_def, out_dcs)
