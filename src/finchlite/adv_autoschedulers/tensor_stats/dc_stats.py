from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from finchlite.finch_logic import Field

from ... import finch_notation as ntn
from ...algebra import Tensor, ffuncs, ftype, int64
from ...algebra.algebra import FinchOperator
from ...compile import BufferizedNDArray, make_extent
from .numeric_stats import NumericStats
from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStatsFactory


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

    from_indices: frozenset[Field]
    to_indices: frozenset[Field]
    value: float


def _int_tuple_ftype(size: int):
    return ftype(tuple(np.int64(0) for _ in range(size)))


class DCStatsFactory(BaseTensorStatsFactory["DCStats"]):
    def __init__(self):
        super().__init__(DCStats)

    def copy_stats(self, stat: DCStats) -> DCStats:
        if not isinstance(stat, DCStats):
            raise TypeError("copy_stats expected a DCStats instance")
        return DCStats.from_def(stat.tensordef.copy(), set(stat.dcs))

    def _mapjoin_union(
        self, new_def: TensorDef, op: FinchOperator, union_args: list[DCStats]
    ) -> DCStats:

        return DCStats._merge_dc_union(new_def, union_args)

    def _mapjoin_join(
        self, new_def: TensorDef, op: FinchOperator, join_args: list[DCStats]
    ) -> DCStats:

        if not join_args:
            return DCStats.from_def(new_def, set())
        join_cover = set().union(*(s.tensordef.index_order for s in join_args))
        if join_cover == set(new_def.index_order):
            return DCStats._merge_dc_join(new_def, join_args)
        return DCStats._merge_dc_union(new_def, join_args)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DCStats,
    ) -> DCStats:
        fields = reduce_indices
        if len(fields) == 0:
            new_def = stats.tensordef.copy()
        else:
            new_def = TensorDef.aggregate(op, init, fields, stats.tensordef)

        dcs = set(stats.dcs) if isinstance(stats, DCStats) else set()
        return DCStats.from_def(new_def, dcs)

    def relabel(self, stats: DCStats, relabel_indices: tuple[Field, ...]) -> DCStats:
        d = stats.tensordef
        new_def = TensorDef.relabel(d, relabel_indices)
        dcs: set[DC] = set(stats.dcs) if isinstance(stats, DCStats) else set()
        return DCStats.from_def(new_def, dcs)

    def reorder(self, stats: DCStats, reorder_indices: tuple[Field, ...]) -> DCStats:
        d = stats.tensordef
        new_def = TensorDef.reorder(d, reorder_indices)
        dcs: set[DC] = set(stats.dcs) if isinstance(stats, DCStats) else set()
        return DCStats.from_def(new_def, dcs)


class DCStats(NumericStats):
    """
    Structural statistics derived from a tensor using degree constraint (DCs).

    DCStats scans a tensor and computes degree constraint (DC) records that
    summarize how index sets relate. These DCs can be used to estimate the
    number of non-fill values without materializing sparse coordinates.
    """

    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        """
        Initialize DCStats from a tensor and its axis names, build the TensorDef,
        and compute degree constraint (DC) records from the tensor’s structure.
        """
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        self.dcs = self._structure_to_dcs(tensor, fields)

    @staticmethod
    def from_def(tensordef: TensorDef, dcs: set[DC]) -> DCStats:
        """
        Build DCStats directly from a TensorDef and an existing DC set.
        """
        self = object.__new__(DCStats)
        self.tensordef = tensordef.copy()
        self.dcs = set(dcs)
        return self

    def _structure_to_dcs(self, arr: Tensor, fields: Iterable[Field]) -> set[DC]:
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
        ndim = arr.ndim

        if ndim == 0:
            return {DC(frozenset(), frozenset(), 1.0)}

        return self._array_to_dcs(arr, fields)

    # Given an arbitrary n-dimensional tensor, we produce 2n+1 degree constraints.
    # For each field i, we compute DC({}, {i}) and DC({i}, {*fields}).
    # Additionally, we compute the nnz for the full tensor DC({}, {*fields}).
    def _array_to_dcs(self, arr: Any, fields: Iterable[Field]) -> set[DC]:
        int64_vector_ftype = BufferizedNDArray.from_numpy(
            np.zeros(1, dtype=np.int64)
        ).ftype
        fields = list(fields)
        ndims = len(fields)
        dim_loop_variables = [ntn.Variable(f"{fields[i]}", int64) for i in range(ndims)]
        dim_array_variables = [
            ntn.Variable(f"x_{fields[i]}", int64_vector_ftype) for i in range(ndims)
        ]
        dim_size_variables = [
            ntn.Variable(f"n_{fields[i]}", int64) for i in range(ndims)
        ]
        dim_array_slots = [
            ntn.Slot(f"x_{fields[i]}_", int64_vector_ftype) for i in range(ndims)
        ]
        dim_proj_variables = [
            ntn.Variable(f"proj_{fields[i]}", int64) for i in range(ndims)
        ]
        dim_dc_variables = [
            ntn.Variable(f"dc_{fields[i]}", int64) for i in range(ndims)
        ]

        A = ntn.Variable("A", arr.ftype)
        A_ = ntn.Slot("A_", arr.ftype)
        A_access = ntn.Unwrap(ntn.Access(A_, ntn.Read(), tuple(dim_loop_variables)))
        A_nnz_variable = ntn.Variable("nnz", int64)

        dim_size_assignments = []
        dim_proj_variable_assigments = []
        dim_dc_variable_assigments = []
        dim_array_unpacks = []
        dim_array_declares = []
        dim_array_increments = []
        for i in range(ndims):
            dim_size_assignments.append(
                ntn.Assign(
                    dim_size_variables[i],
                    ntn.Dimension(A_, ntn.Literal(i)),
                )
            )
            dim_proj_variable_assigments.append(
                ntn.Assign(dim_proj_variables[i], ntn.Literal(int64(0)))
            )
            dim_dc_variable_assigments.append(
                ntn.Assign(dim_dc_variables[i], ntn.Literal(int64(0)))
            )
            dim_array_unpacks.append(
                ntn.Unpack(dim_array_slots[i], dim_array_variables[i])
            )
            dim_array_declares.append(
                ntn.Declare(
                    dim_array_slots[i],
                    ntn.Literal(int64(0)),
                    ntn.Literal(ffuncs.add),
                    (dim_size_variables[i],),
                )
            )
            inc_expr = ntn.Increment(
                ntn.Access(
                    dim_array_slots[i],
                    ntn.Update(ntn.Literal(ffuncs.add)),
                    (dim_loop_variables[i],),
                ),
                ntn.Call(
                    ntn.Literal(ffuncs.ne),
                    (
                        A_access,
                        ntn.Literal(self.tensordef.fill_value),
                    ),
                ),
            )
            dim_array_increments.append(inc_expr)

        array_build_loop: ntn.NotationStatement = ntn.Block(
            (
                *dim_array_increments,
                ntn.Assign(
                    A_nnz_variable,
                    ntn.Call(
                        ntn.Literal(ffuncs.add),
                        (
                            A_nnz_variable,
                            ntn.Call(
                                ntn.Literal(ffuncs.ne),
                                (
                                    A_access,
                                    ntn.Literal(self.tensordef.fill_value),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )
        for i in range(ndims):
            array_build_loop = ntn.Loop(
                dim_loop_variables[i],
                ntn.Call(
                    ntn.Literal(make_extent),
                    (ntn.Literal(int64(0)), dim_size_variables[i]),
                ),
                array_build_loop,
            )

        dim_array_freezes = []
        dc_compute_loops = []
        dim_array_repacks = []
        for i in range(ndims):
            dim_array_freezes.append(
                ntn.Freeze(dim_array_slots[i], ntn.Literal(ffuncs.add))
            )
            dc_compute_loops.append(
                ntn.Loop(
                    dim_loop_variables[i],
                    ntn.Call(
                        ntn.Literal(make_extent),
                        (ntn.Literal(int64(0)), dim_size_variables[i]),
                    ),
                    ntn.Block(
                        (
                            ntn.If(
                                ntn.Call(
                                    ntn.Literal(ffuncs.ne),
                                    (
                                        ntn.Unwrap(
                                            ntn.Access(
                                                dim_array_slots[i],
                                                ntn.Read(),
                                                (dim_loop_variables[i],),
                                            )
                                        ),
                                        ntn.Literal(np.int64(0)),
                                    ),
                                ),
                                ntn.Assign(
                                    dim_proj_variables[i],
                                    ntn.Call(
                                        ntn.Literal(ffuncs.add),
                                        (
                                            dim_proj_variables[i],
                                            ntn.Literal(np.int64(1)),
                                        ),
                                    ),
                                ),
                            ),
                            ntn.Assign(
                                dim_dc_variables[i],
                                ntn.Call(
                                    ntn.Literal(ffuncs.max),
                                    (
                                        dim_dc_variables[i],
                                        ntn.Unwrap(
                                            ntn.Access(
                                                dim_array_slots[i],
                                                ntn.Read(),
                                                (dim_loop_variables[i],),
                                            )
                                        ),
                                    ),
                                ),
                            ),
                        )
                    ),
                )
            )
            dim_array_repacks.append(
                ntn.Repack(dim_array_slots[i], dim_array_variables[i])
            )

        def to_tuple(*args):
            return (*args,)

        dc_args = []
        for i in range(ndims):
            dc_args.append(dim_proj_variables[i])
            dc_args.append(dim_dc_variables[i])
        return_expr = ntn.Return(
            ntn.Call(
                ntn.Literal(to_tuple),
                (*dc_args, A_nnz_variable),
            )
        )

        prgm = ntn.Module(
            (
                ntn.Function(
                    ntn.Variable("array_to_dcs", _int_tuple_ftype(2 * ndims + 1)),
                    (A, *dim_array_variables),
                    ntn.Block(
                        (
                            ntn.Unpack(A_, A),
                            *dim_size_assignments,
                            ntn.Assign(A_nnz_variable, ntn.Literal(int64(0))),
                            *dim_proj_variable_assigments,
                            *dim_dc_variable_assigments,
                            *dim_array_unpacks,
                            *dim_array_declares,
                            array_build_loop,
                            *dim_array_freezes,
                            *dc_compute_loops,
                            *dim_array_repacks,
                            ntn.Repack(A_, A),
                            return_expr,
                        )
                    ),
                ),
            )
        )
        mod = ntn.NotationInterpreter()(prgm)

        dim_array_instances = [
            BufferizedNDArray.from_numpy(np.zeros(arr.shape[i], dtype=np.int64))
            for i in range(ndims)
        ]
        dc_proj_pairs = mod.array_to_dcs(arr, *dim_array_instances)
        dcs = set()
        for i in range(ndims):
            dcs.add(DC(frozenset({}), frozenset({fields[i]}), dc_proj_pairs[2 * i]))
            dcs.add(
                DC(
                    frozenset({fields[i]}),
                    frozenset({*fields}),
                    dc_proj_pairs[2 * i + 1],
                )
            )
        dcs.add(DC(frozenset({}), frozenset({*fields}), dc_proj_pairs[-1]))
        return dcs

    @staticmethod
    def _merge_dc_join(new_def: TensorDef, all_stats: list[DCStats]) -> DCStats:
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

        new_dc: dict[tuple[frozenset[Field], frozenset[Field]], float] = {}
        for stats in all_stats:
            for dc in stats.dcs:
                dc_key = (dc.from_indices, dc.to_indices)
                current_dc = new_dc.get(dc_key, math.inf)
                if dc.value < current_dc:
                    new_dc[dc_key] = dc.value

        new_stats = {DC(X, Y, d) for (X, Y), d in new_dc.items()}
        return DCStats.from_def(new_def, new_stats)

    @staticmethod
    def _merge_dc_union(new_def: TensorDef, all_stats: list[DCStats]) -> DCStats:
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

        dc_keys: Counter[tuple[frozenset[Field], frozenset[Field]]] = Counter()
        stats_dcs: list[dict[tuple[frozenset[Field], frozenset[Field]], float]] = []
        for stats in all_stats:
            dcs: dict[tuple[frozenset[Field], frozenset[Field]], float] = {}
            Z = tuple(
                x for x in new_def.index_order if x not in stats.tensordef.index_order
            )
            # Z = new_def.index_order - stats.tensordef.index_order
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

        new_dcs: dict[tuple[frozenset[Field], frozenset[Field]], float] = {}
        for key, count in dc_keys.items():
            if count == len(all_stats):
                total = sum(d.get(key, 0.0) for d in stats_dcs)
                X, Y = key
                if Y.issubset(new_def.index_order):
                    total = min(total, new_def.get_dim_space_size(Y))
                new_dcs[key] = min(float(2**64), total)

        new_stats = {DC(X, Y, d) for (X, Y), d in new_dcs.items()}
        return DCStats.from_def(new_def, new_stats)

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
        idx: frozenset[Field] = frozenset(self.tensordef.dim_sizes.keys())
        if len(idx) == 0:
            return 1.0

        best: dict[frozenset[Field], float] = {frozenset(): 1.0}
        frontier: set[frozenset[Field]] = {frozenset()}

        while True:
            current_bound = best.get(idx, math.inf)
            new_frontier: set[frozenset[Field]] = set()

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

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        dcs = self.dcs
        dc_embedding = [
            dc.value
            for dc in sorted(
                dcs,
                key=lambda dc: (
                    tuple(sorted(str(f) for f in dc.from_indices)),
                    tuple(sorted(str(f) for f in dc.to_indices)),
                ),
            )
        ]
        return np.log2(np.array([*sizes, *dc_embedding]))
