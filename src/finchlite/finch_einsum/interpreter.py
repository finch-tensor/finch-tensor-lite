import operator

import numpy as np

from ..algebra import Tensor, overwrite, promote_max, promote_min, TensorFType
from . import nodes as ein
from ..symbolic import ftype, gensym


nary_ops = {
    operator.add: "add",
    operator.mul: "multiply",
    operator.sub: "subtract",
    operator.truediv: "divide",
    operator.floordiv: "floor_divide",
    operator.mod: "remainder",
    operator.pow: "power",
    operator.eq: "equal",
    operator.ne: "not_equal",
    operator.lt: "less",
    operator.le: "less_equal",
    operator.gt: "greater",
    operator.ge: "greater_equal",
    operator.and_: "bitwise_and",
    operator.or_: "bitwise_or",
    operator.xor: "bitwise_xor",
    operator.lshift: "bitwise_left_shift",
    operator.rshift: "bitwise_right_shift",
    np.logical_and: "logical_and",
    np.logical_or: "logical_or",
    np.logical_not: "logical_not",
    promote_min: "minimum",
    promote_max: "maximum",
}
unary_ops = {
    operator.pos: "positive",
    operator.neg: "negative",
    operator.invert: "bitwise_invert",
    operator.abs: "absolute",
    np.sqrt: "sqrt",
    np.exp: "exp",
    np.log: "log",
    np.log1p: "log1p",
    np.log10: "log10",
    np.log2: "log2",
    np.sin: "sin",
    np.cos: "cos",
    np.tan: "tan",
    np.sinh: "sinh",
    np.cosh: "cosh",
    np.tanh: "tanh",
    np.arcsin: "arcsin",
    np.arccos: "arccos",
    np.arctan: "arctan",
    np.arcsinh: "arcsinh",
    np.arccosh: "arccosh",
    np.arctanh: "arctanh",
}

reduction_ops = {
    operator.add: "sum",
    operator.mul: "prod",
    operator.and_: "all",
    operator.or_: "any",
    promote_min: "min",
    promote_max: "max",
    np.logical_and: "all",
    np.logical_or: "any",
}


class EinsumInterpreter:
    def __init__(self, xp=None, bindings=None, loops=None):
        if bindings is None:
            bindings = {}
        if xp is None:
            xp = np
        self.bindings = bindings
        self.xp = xp
        self.loops = loops

    def __call__(self, node):
        from ..tensor import (
            SparseTensor,
            SparseTensorFType,
        )

        xp = self.xp
        match node:
            case ein.Literal(val):
                return val
            case ein.Alias(name):
                return self.bindings[name]
            case ein.Call(func, args):
                func = self(func)
                if len(args) == 1:
                    func = getattr(xp, unary_ops[func])
                else:
                    func = getattr(xp, nary_ops[func])
                vals = [self(arg) for arg in args]
                return func(*vals)

            #access a tensor with only indices
            case ein.Access(tns, idxs) if all(isinstance(idx, ein.Index) for idx in idxs):
                assert len(idxs) == len(set(idxs))
                assert self.loops is not None

                #convert named idxs to positional, integer indices
                perm = [idxs.index(idx) for idx in self.loops if idx in idxs]
                
                tns = self(tns) #evaluate the tensor

                #if there are fewer indicies than dimensions, add the remaining dimensions as if they werent permutated
                if hasattr(tns, "ndim") and len(perm) < tns.ndim: 
                    perm = perm + [i for i in range(len(perm), tns.ndim)]

                tns = xp.permute_dims(tns, perm) #permute the dimensions
                return xp.expand_dims(
                    tns,
                    [i for i in range(len(self.loops)) if self.loops[i] not in idxs],
                )

            #access a tensor with only one indirect access index
            case ein.Access(tns, idxs) if len(idxs) == 1:
                idx = self(idxs[0])
                tns = self(tns) #evaluate the tensor
            
                flat_idx = idx if idx.ndim == 1 else xp.ravel_multi_index(idx.T, tns.shape)
                return tns.flat[flat_idx] #return a 1-d array by definition

            #access a tensor with a mixture of indices and other expressions
            case ein.Access(tns, idxs):
                assert self.loops is not None
                true_idxs = node.get_idxs() #true field iteratior indicies
                assert all(isinstance(idx, ein.Index) for idx in true_idxs)

                # evaluate the tensor to access
                tns = self(tns)
                assert len(idxs) == len(tns.shape)

                # Separate ein.Index positions from indirect access positions
                ein_idx_positions = [i for i, idx in enumerate(idxs) if isinstance(idx, ein.Index)]
                indirect_positions = [i for i, idx in enumerate(idxs) if not isinstance(idx, ein.Index)]
                
                # Compute cartesian product for ein.Index instances only
                if ein_idx_positions:
                    ein_idx_ranges = [xp.arange(tns.shape[i]) for i in ein_idx_positions]
                    ein_combo_idxs = xp.meshgrid(*ein_idx_ranges, indexing="ij")
                    ein_combo_idxs = xp.stack(ein_combo_idxs, axis=-1)
                    ein_combo_idxs = ein_combo_idxs.reshape(-1, len(ein_idx_positions))  # Shape: (N_ein, num_ein_indices)
                else:
                    ein_combo_idxs = xp.empty((1, 0), dtype=xp.int64)
                
                # Evaluate indirect accesses as a "super index" (no cartesian product amongst them)
                if indirect_positions:
                    indirect_vals = [self(idxs[i]).flatten() for i in indirect_positions]
                    indirect_combo = xp.stack(indirect_vals, axis=-1)  # Shape: (M_indirect, num_indirect_indices)
                else:
                    indirect_combo = xp.empty((1, 0), dtype=xp.int64)
                
                # Compute cartesian product between ein.Index group and indirect group
                # The indirect group is treated as a single "super index" (no cartesian product amongst indirect values)
                n_ein = ein_combo_idxs.shape[0]
                n_indirect = indirect_combo.shape[0]
                
                # Determine which group comes first/last to decide iteration order
                # For standard row-major order, the first dimension varies slowest
                # Check if ein or indirect comes first
                first_ein_pos = ein_idx_positions[0] if ein_idx_positions else float('inf')
                first_indirect_pos = indirect_positions[0] if indirect_positions else float('inf')
                
                if first_ein_pos < first_indirect_pos:
                    # ein comes first: repeat ein, tile indirect (ein varies slowest)
                    ein_result = xp.repeat(ein_combo_idxs, n_indirect, axis=0)
                    indirect_result = xp.tile(indirect_combo, (n_ein, 1))
                else:
                    # indirect comes first: repeat indirect, tile ein (indirect varies slowest)
                    indirect_result = xp.repeat(indirect_combo, n_ein, axis=0)
                    ein_result = xp.tile(ein_combo_idxs, (n_indirect, 1))
                
                # Now we need to interleave these back in the original order
                combo_idxs = xp.empty((n_ein * n_indirect, len(idxs)), dtype=xp.int64)
                idx_sizes = []  # Track the size of each index dimension
                for i, idx in enumerate(idxs):
                    if isinstance(idx, ein.Index):
                        pos_in_ein = ein_idx_positions.index(i)
                        combo_idxs[:, i] = ein_result[:, pos_in_ein]
                        idx_sizes.append(tns.shape[i])
                    else:
                        pos_in_indirect = indirect_positions.index(i)
                        combo_idxs[:, i] = indirect_result[:, pos_in_indirect]
                        idx_sizes.append(n_indirect)
                
                # evaluate the output tensor as a flat array
                flat_idx = xp.ravel_multi_index(combo_idxs.T, tns.shape)
                tns = xp.take(tns, flat_idx)

                #calculate child idxs, idxs computed using the parent "true idxs"
                child_idxs = { 
                    parent_idx: [
                        child_idx for child_idx in idxs 
                        if (parent_idx in child_idx.get_idxs())
                    ] for parent_idx in true_idxs 
                }

                # we assert that all the indirect access indicies from the parent idxs have the same size         
                child_idxs_size = {
                    parent_idx: [idx_sizes[idxs.index(child_idx)] for child_idx in child_idxs[parent_idx]]
                    for parent_idx in true_idxs
                }
                assert all(
                    child_idxs_size[parent_idx].count(child_idxs_size[parent_idx][0]) == len(child_idxs_size[parent_idx]) 
                    for parent_idx in true_idxs
                )

                # a mapping from each idx to its axis wrt to current shape
                idxs_axis = {idx: i for i, idx in enumerate(idxs)}

                true_idxs = list(true_idxs)
                true_idxs = sorted(true_idxs, key=lambda idx: idxs_axis[child_idxs[idx][0]])
                print(true_idxs)

                # calculate the final shape of the tensor
                # we merge the child idxs to get the final shape that matches the true idxs
                final_shape = tuple(
                    idx_sizes[idxs.index(child_idxs[parent_idx][0])] 
                    for parent_idx in true_idxs
                )
                tns = tns.reshape(final_shape)

                # permute and broadcast the tensor to be compatible with rest of expression
                perm = [true_idxs.index(idx) for idx in self.loops if idx in true_idxs]
                tns = xp.permute_dims(tns, perm)
                return xp.expand_dims(
                    tns, 
                    [i for i in range(len(self.loops)) if self.loops[i] not in true_idxs]
                )

            case ein.Plan(bodies):
                res = None
                for body in bodies:
                    res = self(body)
                return res
            case ein.Produces(args):
                return tuple(self(arg) for arg in args)

            #get non-zero elements/data array of a sparse tensor
            case ein.GetAttribute(obj, ein.Literal("elems"), _):
                obj = self(obj)
                assert isinstance(ftype(obj), SparseTensorFType)
                assert isinstance(obj, SparseTensor)
                return obj.data 
            #get coord array of a sparse tensor
            case ein.GetAttribute(obj, ein.Literal("coords"), dim):
                obj = self(obj)
                assert isinstance(ftype(obj), SparseTensorFType)
                assert isinstance(obj, SparseTensor)

                # return the coord array for the given dimension or all dimensions
                toReturn = obj.coords if dim is None else obj.coords[:, dim]
                return toReturn
            # gets the shape of a sparse tensor at a given dimension
            case ein.GetAttribute(obj, ein.Literal("shape"), dim):
                obj = self(obj)
                assert isinstance(ftype(obj), SparseTensorFType)
                assert isinstance(obj, SparseTensor)
                assert dim is not None

                # return the shape for the given dimension
                return obj.shape[dim]

            # standard einsum
            case ein.Einsum(op, ein.Alias(tns), idxs, arg) if all(isinstance(idx, ein.Index) for idx in idxs):
                # This is the main entry point for einsum execution
                loops = arg.get_idxs()
                
                assert set(idxs).issubset(loops)
                loops = sorted(loops, key=lambda x: x.name)
                ctx = EinsumInterpreter(self.xp, self.bindings, loops)
                arg = ctx(arg)
                axis = tuple(i for i in range(len(loops)) if loops[i] not in idxs)
                op = self(op)
                if op != overwrite:
                    op = getattr(xp, reduction_ops[op])
                    val = op(arg, axis=axis)
                else:
                    assert set(idxs) == set(loops)
                    val = arg
                dropped = [idx for idx in loops if idx in idxs]
                axis = [dropped.index(idx) for idx in idxs]
                self.bindings[tns] = xp.permute_dims(val, axis)
                return (tns,)

            # indirect einsum
            case ein.Einsum(op, ein.Alias(tns), idxs, arg):
                raise NotImplementedError("Indirect einsum assignment is not implemented")
            case _:
                raise ValueError(f"Unknown einsum type: {type(node)}")
