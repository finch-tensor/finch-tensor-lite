"""
SuitableRep is a class that analyzes query expressions and predicts the representation of results.

Corresponds to the SuitableRep struct in Finch.jl.
"""

from typing import Any, Dict, List
from finchlite.finch_logic.nodes import Alias, Literal, Table, MapJoin, Aggregate, Reorder, Relabel
from finchlite.galley.PhysicalOptimizer.representation import Representation, ElementData
from finchlite.galley.PhysicalOptimizer.rep_operations import *

class SuitableRep:
    def __init__(self, bindings: Dict[Any, Representation] = {}):
        self.bindings: Dict[Any, Representation] = bindings if bindings else {}

    def __call__(self, ex: Any) -> Representation:
        """
        Predict the representation of the result of the query expression.
        """
        if isinstance(ex, Alias):
            return self.bindings[ex]
        elif isinstance(ex, Table):
            if isinstance(ex.tns, Literal):
                return data_rep(ex.tns.val)
            elif hasattr(ex.tns, 'type_'):
                return data_rep(ex.tns.type_)
        elif isinstance(ex, MapJoin):
            return self._handle_mapjoin(ex)
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

    def _handle_mapjoin(self, ex: Any) -> Representation:
        """
        Handle a mapjoin expression.
        Gets reps for all args, expands dims, applies map_rep to predict sparsity, then drops dimensions.
        """
        op = ex.op.val
        args = ex.args
        idxs = ex.fields()

        reps = []
        for arg in args:
            rep = self(arg)
            arg_fields = self._get_fields(arg)
            dims = [i for i, idx in enumerate(idxs, 1) if idx not in arg_fields]
            if dims:
                rep = expanddims_rep(rep, dims)
            reps.append(rep)
        
        return map_rep(op, *reps)
    
    def _handle_aggregate(self, ex: Any) -> Representation:
        """
        Handle aggregate expression.
        Gets rep, and calls aggregate_rep to predict sparsity.
        """
        idxs = self._get_fields(ex.arg)
        arg_rep = self(ex.arg)
        reduction_dims = [
            idxs.index(idx) + 1 for idx in ex.idxs if idx in idxs
        ]

        return aggregate_rep(ex.op.val, ex.init.val, arg_rep, reduction_dims)
    
    def _handle_reorder(self, ex: Any) -> Representation:
        """
        Handle reorder expression.
        Gets rep, drops dimensions, permutes remaining dimensions, and adds new dimensions.
        """
        rep = self(ex.arg)
        idxs = self._get_fields(ex.arg)
        reduction_dims = [
            i + 1 for i, idx in enumerate(idxs) if idx not in ex.idxs
        ]
        if reduction_dims:
            rep = dropdims_rep(rep, reduction_dims)
        
        rem = [idx for idx in idxs if idx in ex.idxs]
        if rem:
            perm = sorted(range(len(rem)), key=lambda i: ex.idxs.index(rem[i]))
            rep = permutedims_rep(rep, perm)
        new_dims = [i + 1 for i, idx in enumerate(ex.idxs) if idx not in idxs]
        if new_dims:
            rep = expanddims_rep(rep, new_dims)
        return rep
    
    def _get_fields(self, ex: Any) -> List[Any]:
        """
        Get the fields of the expression.
        """
        return ex.fields()
