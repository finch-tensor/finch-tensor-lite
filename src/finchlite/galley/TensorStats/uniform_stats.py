from collections.abc import Callable
from typing import Any, Self

from finchlite.finch_logic import Field
from finchlite.algebra.algebra import is_annihilator, is_identity
from .tensor_def import TensorDef
from .tensor_stats import TensorStats
import numpy as np
import math

class UniformStats(TensorStats):
    nnz : float
    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        val = tensor
        
        if hasattr(val, "tns"):
            val = val.tns.val
        if hasattr(val, "val") and not hasattr(val, "to_numpy"):
            val = val.val

        if hasattr(val, "to_numpy"):
            arr = val.to_numpy()
            self.nnz = float(np.sum(arr != self.tensordef.fill_value))
        else:
            self.nnz = float(self._get_volume(self.tensordef))

    @classmethod
    def from_def(cls, d: TensorDef, nnz : float | None = None) -> Self:
        us = object.__new__(cls)
        us.tensordef = d.copy()
        if nnz is None:
            us.nnz = float(cls._get_volume(d))
        else :
            us.nnz = float(nnz)
        return us
    
    @staticmethod
    def _get_volume(d:TensorDef)->float:
        vol = 1.0
        for size in d.dim_sizes.values():
            vol *= size 
        return vol

    @staticmethod
    def copy_stats(stat: TensorStats) -> TensorStats:
        """
        Deep copy of a UniformStats object.
        """
        if not isinstance(stat, UniformStats):
            raise TypeError("copy_stats expected a UniformStats instance")
        return UniformStats.from_def(stat.tensordef.copy(),stat.nnz)

    def estimate_non_fill_values(self) -> float:
        return self.nnz

    @staticmethod
    def mapjoin(op: Callable, *args: TensorStats) -> TensorStats:
        def_args = [stat.tensordef for stat in args]
        new_def = TensorDef.mapjoin(op, *def_args)
        new_vol = UniformStats._get_volume(new_def)

        if new_vol == 0.0 :
            return UniformStats.from_def(new_def,0.0)
        
        join_probs :list[float] = []
        union_probs :list[float] = []

        for s in args:
            vol = UniformStats._get_volume(s.tensordef)
            p = s.estimate_non_fill_values()/ vol if vol > 0 else 0.0

            if is_annihilator(op,s.tensordef.fill_value):
                join_probs.append(p)
            else :
                union_probs.append(p)

        res_p = 1.0
        if join_probs:
            for p in join_probs:
                res_p *= p
        elif union_probs:
            inv_p = 1.0
            for p in union_probs:
                inv_p *= (1-p)
            res_p = 1 - inv_p
        else :
            res_p = 1.0

        return UniformStats.from_def(new_def,res_p*new_vol)

    @staticmethod
    def aggregate(
        op: Callable[..., Any],
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: "TensorStats",
    ) -> "UniformStats":
        new_def = TensorDef.aggregate(op, init, reduce_indices, stats.tensordef)
        res_vol = UniformStats._get_volume(new_def)
        red_set = set(reduce_indices) & set(stats.tensordef.index_order)
        k = math.prod(int(stats.tensordef.dim_sizes[x]) for x in red_set)
        old_vol = UniformStats._get_volume(stats.tensordef)
        p_old = stats.estimate_non_fill_values()/old_vol if old_vol>0 else 0.0

        if is_annihilator(op,stats.tensordef.fill_value):
            res_p = math.pow(p_old,k)
        elif is_identity(op,stats.tensordef.fill_value):
            res_p = 1 - math.pow((1-p_old),k)
        else :
            res_p = 1.0

        return UniformStats.from_def(new_def,res_p*res_vol)


    @staticmethod
    def issimilar(a: TensorStats, b: TensorStats) -> bool:
        return (
            isinstance(a, UniformStats)
            and isinstance(b, UniformStats)
            and a.dim_sizes == b.dim_sizes
            and a.fill_value == b.fill_value
            #should I add a case for nnz ?
            and math.isclose(a.nnz, b.nnz, rel_tol=1e-9)
        )

    @staticmethod
    def relabel(
        stats: "TensorStats", relabel_indices: tuple[Field, ...]
    ) -> "UniformStats":

        d = stats.tensordef
        new_def = TensorDef.relabel(d, relabel_indices)
        return UniformStats.from_def(new_def,stats.estimate_non_fill_values())

    @staticmethod
    def reorder(
        stats: "TensorStats", reorder_indices: tuple[Field, ...]
    ) -> "UniformStats":

        d = stats.tensordef
        new_def = TensorDef.reorder(d, reorder_indices)
        return UniformStats.from_def(new_def, stats.estimate_non_fill_values())
