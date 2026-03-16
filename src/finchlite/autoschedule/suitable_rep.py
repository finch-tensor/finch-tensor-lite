"""
SuitableRep analyzes query expressions and predicts the representation
of results.

Corresponds to the SuitableRep struct in Finch.jl.
"""

import logging
from abc import abstractmethod
from typing import Any

import numpy as np

from finchlite.finch_logic.nodes import (
    Aggregate,
    Alias,
    Literal,
    MapJoin,
    Relabel,
    Reorder,
    Table,
)

from .. import finch_logic as lgc
from ..algebra import TensorFType
from ..finch_assembly import AssemblyLibrary
from ..finch_logic import LogicLoader, MockLogicLoader
from ..util.logging import LOG_LOGIC_POST_OPT
from .formatter import LogicFormatter
from .rep_operations import (
    aggregate_rep,
    data_rep,
    dropdims_rep,
    expanddims_rep,
    map_rep,
    permutedims_rep,
)
from .representation import (
    DenseData,
    ElementData,
    ExtrudeData,
    HollowData,
    RepeatData,
    SparseData,
    SparseRepeatData,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)

Representation = (
    ElementData
    | DenseData
    | ExtrudeData
    | HollowData
    | RepeatData
    | SparseData
    | SparseRepeatData
)


def toposort(chains: list[list[Any]]) -> list[Any] | None:
    """
    Does topological sort on a list of chains.
    Returns the sorted list, or None if there's a cycle.
    """
    chains = [list(chain) for chain in chains if chain]
    if not chains:
        return []

    parents: dict[Any, int] = {}
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
    def __init__(self, bindings: dict[Any, Representation] | None = None):
        self.bindings: dict[Any, Representation] = bindings if bindings else {}

    def __call__(self, ex: Any) -> Representation:
        """
        Predict the representation of the result of the query expression.
        """
        match ex:
            case Alias():
                return self.bindings[ex]
            case Table():
                if isinstance(ex.tns, Literal):
                    return data_rep(ex.tns.val)
                if hasattr(ex.tns, "type_"):
                    return data_rep(ex.tns.type_)
                raise ValueError(f"bad table type: {type(ex)}")
            case Reorder() if isinstance(ex.arg, MapJoin):
                return self._handle_reorder_mapjoin(ex)
            case Aggregate():
                return self._handle_aggregate(ex)
            case Reorder():
                return self._handle_reorder(ex)
            case Relabel():
                return self(ex.arg)
            case Literal():
                return ElementData(ex.val, type(ex.val))
            case _:
                raise ValueError(f"Unrecognized expression kind: {type(ex)}")

    def _handle_reorder_mapjoin(self, ex: Reorder) -> Representation:
        """
        Handle reorder(mapjoin())
        Computes canonical ordering of all indices, expands args, applies
        map_rep, then drops extra dims.
        """
        mapjoin = ex.arg
        assert isinstance(mapjoin, MapJoin)
        idxs = list(ex.idxs)

        arg_fields_list = [list(arg.fields()) for arg in mapjoin.args]
        idxs_2 = toposort(arg_fields_list + [idxs])
        if idxs_2 is None:
            raise ValueError("Cycle detected in toposort")

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


class SmartLogicFormatter(LogicFormatter):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.loader = loader

    @abstractmethod
    def get_output_tns_type(
        self, fill_value: Any, shape_type: tuple[Any, ...], rep: Representation
    ): ...

    def __call__(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        dict[lgc.Alias, tuple[lgc.Field | None, ...]],
    ]:
        bindings = bindings.copy()
        suitable_rep = SuitableRep(
            bindings={alias: data_rep(ftype) for alias, ftype in bindings.items()}
        )
        fill_values = prgm.infer_fill_value(
            {var: val.fill_value for var, val in bindings.items()}
        )
        shape_types = prgm.infer_shape_type(
            {var: val.shape_type for var, val in bindings.items()}
        )

        def formatter(node: lgc.LogicStatement):
            match node:
                case lgc.Plan(bodies):
                    for body in bodies:
                        formatter(body)
                case lgc.Query(lhs, rhs):
                    if lhs not in bindings:
                        rep = suitable_rep(rhs)
                        shape_type = tuple(
                            dim if dim is not None else np.intp
                            for dim in shape_types[lhs]
                        )
                        tns = self.get_output_tns_type(
                            fill_values[lhs], shape_type, rep
                        )

                        bindings[lhs] = tns
                        suitable_rep.bindings[lhs] = rep
                case lgc.Produces(_):
                    pass
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        formatter(prgm)

        logger.debug(prgm)

        lib, bindings, shape_vars = self.loader(prgm, bindings)
        return lib, bindings, shape_vars
