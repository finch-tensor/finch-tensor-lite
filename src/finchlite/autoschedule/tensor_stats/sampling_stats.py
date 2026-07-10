from __future__ import annotations

import math
from typing import Any
import numpy as np
from finchlite.algebra.algebra import FinchOperator, is_annihilator, is_identity
from finchlite.finch_logic import Field
from .numeric_stats import NumericStats
from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStatsFactory


def _duj1(d_n:float,f_1:float,q:float,n:float)->float:
    """"
    Using un-smoothened first order jackknife estimator
    D_uj1 = (1-(1-q)*f_1/n)^{-1} * d_n
    """
    if d_n == 0:
        return 0.0
    denom = 1 - ((1-q)*f_1)/max(n,1.0)
    if denom<=0:
        return d_n
    return d_n/denom


def _outer_multiply(a:np.ndarray,a_order:list,b:np.ndarray,b_order:list)->tuple[np.ndarray,list]:
    "multiplying sketches and keeping their index order"
    combined_order = list(a_order)
    for f in b_order:
        if f not in combined_order:
            combined_order.append(f)
    
    a_exp = a
    a_cur = list(a_order)
    for i,f in enumerate(combined_order):
        if f not in a_cur:
            a_exp = np.expand_dims(a_exp,axis=i)
            a_cur.insert(i,f)

    b_exp = b
    b_cur = list(b_order)
    for i,f in enumerate(combined_order):
        if f not in b_cur:
            b_exp = np.expand_dims(b_exp,axis=i)
            b_cur.insert(i,f)
    
    shape = tuple(
        max(a_exp.shape[combined_order.index(f)],b_exp.shape[combined_order.index(f)]
            ) for f in combined_order
    )

    a_exp = np.broadcast_to(a_exp,shape).copy()
    b_exp = np.broadcast_to(b_exp,shape).copy()

    return a_exp*b_exp, combined_order
    
def _expand_sketch_to(sketch:np.ndarray,current_order:list,target_order:list,new_def:TensorDef)->np.ndarray:
    result = sketch
    current = list(current_order)
    for i,f in enumerate(target_order):
        if f not in current:
            result = np.expand_dims(result,axis=i)
            current.insert(i,f)
    target_shape = tuple(int(new_def.dim_sizes[f]) for f in target_order)
    return np.broadcast_to(result,target_shape).copy()

def _reorder_to(sketch:np.ndarray,current_order:list,target_order:list)->np.ndarray:
    if current_order ==target_order:
        return sketch
    perm = [current_order.index(f) for f in target_order if f in current_order]
    if len(perm) != len(current_order):
        return sketch
    return np.transpose(sketch,perm)


class SamplingStatsFactory(BaseTensorStatsFactory["SamplingStats"]):

    def __init__(self, sample_prob:float=0.5):
        super().__init__(SamplingStats)
        self.sample_prob = sample_prob
        self._masks : dict = {}
        self._rng = np.random.default_rng()

    def _get_mask(self,field:Field,size:int)->np.ndarray:
        "Returns mask for dimension that already exists or creates a new one"
        mask_key = (field,size)
        if mask_key not in self._masks:
            self._masks[mask_key]=(self._rng.random(size)<self.sample_prob).astype(float)
        return self._masks[mask_key]

    def __call__(self, tensor:Any, fields:tuple[Field,...])->SamplingStats:
        return SamplingStats(tensor,fields,sample_prob=self.sample_prob,mask_fn=self._get_mask)
    
    def copy_stats(self,stat:SamplingStats)->SamplingStats:
        if not isinstance(stat,SamplingStats):
            raise TypeError("copy_stats expected a SamplingStats instance")
        return SamplingStats.from_def(stat.tensordef.copy(),stat.sketch.copy(),
                                      set(stat.remainder_dims),stat.sample_prob,dict(stat.remainder_dim_sizes))
    
    def _mapjoin_join(self, new_def:TensorDef, op:FinchOperator, join_args:list[SamplingStats]):
        """
        N(C)_i =  N(A)_j * N(B)_k 
        """

        if len(join_args)==1:
            return self.copy_stats(join_args[0])
        
        result = join_args[0].sketch.copy()
        result_order = list(join_args[0].tensordef.index_order)

        for arg in join_args[1:]:
            result,result_order = _outer_multiply(result,result_order,arg.sketch,list(arg.tensordef.index_order))

        new_remainder: set[Field] = set()
        new_remainder_sizes : dict = {}
        for arg in join_args:

            new_remainder |= arg.remainder_dims
            new_remainder_sizes.update(arg.remainder_dim_sizes)
        
        result = _reorder_to(result,result_order,list(new_def.index_order))

        return SamplingStats.from_def(new_def,result,new_remainder,self.sample_prob,new_remainder_sizes)
    
    def _mapjoin_union(self, new_def:TensorDef, op:FinchOperator, union_args:list[SamplingStats]):
        """
        N(C)_i = sum_{juk\\i}[N(A)_j *prod_{l in k\\j}n(B)_l + N(B)_k *prod_{l in j\\k}n(A)_l ] - N(A)_j*N(B)_k 
        """
        output_indices = set(new_def.index_order)
        result_shape = tuple(int(new_def.dim_sizes[f]) for f in new_def.index_order)
        result = np.zeros(result_shape,dtype=float)
        new_remainder : set[Field] = set()
        new_remainder_sizes : dict = {}

        for arg in union_args:
            new_remainder_sizes.update(arg.remainder_dim_sizes)
            arg_indices = set(arg.tensordef.index_order)
            other_free_size = 1.0
            for other in union_args:
                if other is arg:
                    continue
                for f in other.tensordef.index_order:
                    if f not in arg_indices and f not in output_indices:
                        other_free_size *= other.tensordef.dim_sizes.get(f,1.0)
                for f in other.remainder_dims:
                    other_free_size *= other.tensordef.dim_sizes.get(f,1.0)
            
            expanded = _expand_sketch_to(arg.sketch,list(arg.tensordef.index_order),
                                         list(new_def.index_order),new_def,)
            
            result = result + expanded*other_free_size 
            new_remainder |= arg.remainder_dims

        if len(union_args)>=2:
            inter = union_args[0].sketch.copy()
            inter_order = list(union_args[0].tensordef.index_order)
            for arg in union_args[1:]:
                inter,inter_order = _outer_multiply(inter,inter_order,arg.sketch,list(arg.tensordef.index_order),)
            
            inter_expanded = _expand_sketch_to(inter,inter_order,list(new_def.index_order),
                                               new_def,)
            
            result = result - inter_expanded

        return SamplingStats.from_def(new_def,result,new_remainder,self.sample_prob,new_remainder_sizes)

    def aggregate(self, op:FinchOperator, init: Any| None, reduce_indices : tuple[Field,...], stats:SamplingStats):
        """
        op is identity on fill : N(B)_i = sum_j N(A)_j 
        op annihilates fill: N(B)_i = prod(l in k)n_l * min_k 1[N(A)_j > 0]
        otherwise : N(B)_i = prod(l in k)n_l * exists(N(A)_j)
        """

        new_def = TensorDef.aggregate(op,init,reduce_indices,stats.tensordef)
        reduce_set = set(reduce_indices) & set(stats.tensordef.index_order)
        index_order = list(stats.tensordef.index_order)
        reduce_axes = tuple(
            index_order.index(f) for f in index_order if f in reduce_set
        )
        #check is_annihilator 
        if is_identity(op,stats.tensordef.fill_value):
            new_sketch = stats.sketch.copy()
            for axis in sorted(reduce_axes,reverse=True):
                new_sketch = np.sum(new_sketch,axis = axis)
        elif is_annihilator(op,stats.tensordef.fill_value):
            exists = (stats.sketch>0).astype(float)#why > 0 ?
            for axis in sorted(reduce_axes,reverse=True):
                exists = np.min(exists,axis=axis)
            new_sketch = exists * math.prod(int(stats.tensordef.dim_sizes[f]) for f in reduce_set)
        else :
            prod_n = math.prod(int(stats.tensordef.dim_sizes[f]) for f in reduce_set)
            exists = (stats.sketch>0).astype(float)
            for axis in sorted(reduce_axes,reverse=True):
                exists = np.max(exists,axis=axis)
            new_sketch = prod_n*exists 
        
        new_remainder = stats.remainder_dims | reduce_set
        new_remainder_sizes = dict(stats.remainder_dim_sizes)
        for f in reduce_set:
            new_remainder_sizes[f] = stats.tensordef.dim_sizes[f]
        return SamplingStats.from_def(new_def,new_sketch,new_remainder,stats.sample_prob,new_remainder_sizes)

    def relabel(self, stats:SamplingStats, relabel_indices:tuple[Field,...])->SamplingStats:
        new_def = TensorDef.relabel(stats.tensordef,relabel_indices)
        return SamplingStats.from_def(new_def,stats.sketch.copy(),set(stats.remainder_dims),stats.sample_prob,dict(stats.remainder_dim_sizes))

    def reorder(self,stats:SamplingStats,reorder_indices:tuple[Field,...])->SamplingStats:
        new_def = TensorDef.reorder(stats.tensordef,reorder_indices)
        return SamplingStats.from_def(new_def,stats.sketch.copy(),set(stats.remainder_dims),stats.sample_prob,dict(stats.remainder_dim_sizes))


class SamplingStats(NumericStats):
    """
    sketch : numpy array over bound dimension
    remainder_dims : 'free' dimension -> absent in the output
    sample_prob : Bernoulli sample prob
    """

    sketch : np.ndarray
    remainder_dims : set
    sample_prob : float

    def __init__(self, tensor:Any, fields:tuple[Field,...],sample_prob:float=0.5,mask_fn=None):
        self.tensordef = TensorDef.from_tensor(tensor,fields)
        self.sample_prob = sample_prob
        self.remainder_dims = set()
        self.remainder_dim_sizes : dict = {}

        val = tensor
        if hasattr(val,"tns"):
            val = val.tns.val
        if hasattr(val,"val") and not hasattr(val,"to_numpy"):
            val = val.val
        if hasattr(val,"to_numpy"):
            arr = val.to_numpy()
        else :
            shape = tuple(int(self.tensordef.dim_sizes[f]) for f in fields)
            arr = np.zeros(shape,dtype=float)
        
        fill = self.tensordef.fill_value

        #defining one Bernoulli mask per dimension, an entry will survive only if all its indices are kept
        #masks has the 0's 1's combination for each entry in a dimension
        masks = []
        for field in fields:
            size = int(self.tensordef.dim_sizes[field])
            if mask_fn is not None:
                mask = mask_fn(field,size)
            else :
                mask = (np.random.default_rng().random(size)<sample_prob).astype(float)
            masks.append(mask)
        
        #combining the masks to create a filter to sample 
        combined = masks[0]
        for mask in masks[1:]:
            combined = np.multiply.outer(combined,mask)
        
        #keeping the nnz in the tensor intact
        #sketch is creating the sample with the filter over the tensor
        non_fill = (arr!=fill).astype(float)
        self.sketch = non_fill*combined

    @classmethod
    def from_def(cls,d:TensorDef,sketch:np.ndarray,remainder_dims:set,sample_prob:float, remainder_dim_sizes: dict| None=None)->SamplingStats:
        ss = object.__new__(cls)
        ss.tensordef = d.copy()
        ss.sketch = sketch.copy()
        ss.remainder_dims = set(remainder_dims)
        ss.sample_prob = sample_prob
        ss.remainder_dim_sizes = dict(remainder_dim_sizes) if remainder_dim_sizes else {}
        return ss
    
    def estimate_non_fill_values(self)->float:
        """"
        Using un-smoothened first order jackknife estimator
        D_uj1 = (1-(1-q)*f_1/n)^{-1} * d_n

        d_n : positions with sketch count > 0 (distinct positions observed in the sample)
        f_1 : positions with sketch count = 1 (seen exactly once)
        n = total sample size
        N = population size [Total without sampling]
        q = n/N
        """

        flat = self.sketch.flatten()
        d_n = float(np.sum(flat>0))
        f_1 = float(np.sum(flat==1))
        n = float(np.sum(flat))
        bound_size = math.prod(int(self.tensordef.dim_sizes[f]) for f in self.tensordef.index_order
                               ) if self.tensordef.index_order else 1
        remainder_size = math.prod(int(self.remainder_dim_sizes.get(f,1)) for f in self.remainder_dims
                                   ) if self.remainder_dims else 1
        N = bound_size*remainder_size
        all_dims = list(self.tensordef.index_order) + list(self.remainder_dims)
        ndims = len(all_dims)
        q = self.sample_prob ** ndims

        return _duj1(d_n,f_1,q,n)
    
    def get_embedding(self)->np.ndarray:
        sizes = [float(self.tensordef.dim_sizes[f]) for f in self.tensordef.index_order]
        nnz = self.estimate_non_fill_values()
        size_part = np.log2(np.array(sizes))
        nnz_part = np.log2(np.array(nnz)+1)
        return np.concatenate([size_part,nnz_part])

        


