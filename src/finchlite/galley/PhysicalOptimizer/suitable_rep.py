"""
SuitableRep is a class that analyzes query expressions and predicts the representation of results.

Corresponds to the SuitableRep struct in Finch.jl.
"""

from typing import Any, Union

from finchlite.finch_logic.nodes import (
    Aggregate,
    Alias,
    Literal,
    MapJoin,
    Relabel,
    Reorder,
    Table,
)
from finchlite.galley.PhysicalOptimizer.rep_operations import *
from finchlite.galley.PhysicalOptimizer.representation import (
    DenseData,
    ElementData,
    ExtrudeData,
    HollowData,
    RepeatData,
    SparseData,
)

Representation = Union[
    ElementData, DenseData, ExtrudeData, HollowData, RepeatData, SparseData
]


def toposort(chains: list[list[Any]]) -> list[Any] | None:
    """
    Does topological sort on a list of chains.
    Returns the sorted list, or None if there's a cycle.
    """
    chains = [list(chain) for chain in chains if chain]
    if not chains:
        return []

    parents = {}
    for chain in chains:
        for i, node in enumerate(chain):
            if i == 0:
                parents.setdefault(node, 0)
            else:
                parents[node] = parents.get(node, 0) + 1

    roots = [node for node in parents if parents[node] == 0]
    sorted_nodes = []

    while parents:
        if not roots:
            return None
        node = roots.pop()
        sorted_nodes.append(node)
        for chain in chains:
            if chain and chain[0] == node:
                chain.pop(0)
                if chain:
                    parents[chain[0]] -= 1
                    if parents[chain[0]] == 0:
                        roots.append(chain[0])
        parents.pop(node)
    return sorted_nodes


class SuitableRep:
    def __init__(self, bindings: dict[Any, Representation] = {}):
        self.bindings: dict[Any, Representation] = bindings if bindings else {}

    def __call__(self, ex: Any) -> Representation:
        """
        Predict the representation of the result of the query expression.
        """
        if isinstance(ex, Alias):
            return self.bindings[ex]
        if isinstance(ex, Table):
            if isinstance(ex.tns, Literal):
                return data_rep(ex.tns.val)
            if hasattr(ex.tns, "type_"):
                return data_rep(ex.tns.type_)
        elif isinstance(ex, Reorder) and isinstance(ex.arg, MapJoin):
            return self._handle_reorder_mapjoin(ex)
        elif isinstance(ex, Aggregate):
            return self._handle_aggregate(ex)
        elif isinstance(ex, Reorder):
            return self._handle_reorder(ex)
        elif isinstance(ex, Relabel):
            return self(ex.arg)
        elif isinstance(ex, Literal):
            return ElementData(ex.val, type(ex.val))
        else:
            raise ValueError(f"Unrecognized expression kind: {type(ex)}")

    def _handle_reorder_mapjoin(self, ex: Reorder) -> Representation:
        """
        Handle reorder(mapjoin())
        Computes canonical ordering of all indices, expands args, applies map_rep, then drops extra dims.
        """
        mapjoin = ex.arg
        idxs = list(ex.idxs)

        arg_fields_list = [list(arg.fields()) for arg in mapjoin.args]
        idxs_2 = toposort(arg_fields_list + [idxs])

        reps = []
        for arg in mapjoin.args:
            rep = self(arg)
            arg_fields = list(arg.fields())

            dims = []
            for i, idx in enumerate(idxs_2):
                if idx not in arg_fields:
                    dims.append(i + 1)

            if dims:
                rep = expanddims_rep(rep, dims)
            reps.append(rep)

        rep = map_rep(mapjoin.op.val, *reps)

        dims_to_drop = []
        for i, idx in enumerate(idxs_2):
            if idx not in idxs:
                dims_to_drop.append(i + 1)

        if dims_to_drop:
            rep = dropdims_rep(rep, dims_to_drop)

        return rep

    def _handle_aggregate(self, ex: Any) -> Representation:
        """
        Handle aggregate expression.
        Gets rep, and calls aggregate_rep to predict sparsity.
        """
        idxs = list(ex.arg.fields())
        arg_rep = self(ex.arg)

        reduction_dims = []
        for idx in ex.idxs:
            dim_num = idxs.index(idx) + 1
            reduction_dims.append(dim_num)

        return aggregate_rep(ex.op.val, ex.init.val, arg_rep, reduction_dims)

    def _handle_reorder(self, ex: Reorder) -> Representation:
        """
        Handle reorder expression.
        drop dims not in target, permute remaining dims, expand new dims.
        """
        rep = self(ex.arg)
        idxs = list(ex.arg.fields())

        dims_to_drop = []
        for i, idx in enumerate(idxs):
            if idx not in ex.idxs:
                dims_to_drop.append(i + 1)
        if dims_to_drop:
            rep = dropdims_rep(rep, dims_to_drop)

        intersection = [idx for idx in idxs if idx in ex.idxs]
        if intersection:
            perm = []
            for i in range(len(intersection)):
                target_pos = ex.idxs.index(intersection[i])
                perm.append(target_pos)

            sorted_indices = sorted(range(len(perm)), key=lambda i: perm[i])
            rep = permutedims_rep(rep, sorted_indices)

        dims_to_expand = []
        for i, idx in enumerate(ex.idxs):
            if idx not in idxs:
                dims_to_expand.append(i + 1)
        if dims_to_expand:
            rep = expanddims_rep(rep, dims_to_expand)

        return rep
