import operator

import numpy as np

from ..algebra import overwrite, promote_max, promote_min
from ..symbolic import ftype
from . import nodes as ein

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

            # access a tensor with only indices
            case ein.Access(tns, idxs) if all(
                isinstance(idx, ein.Index) for idx in idxs
            ):
                assert len(idxs) == len(set(idxs))
                assert self.loops is not None

                # convert named idxs to positional, integer indices
                perm = [idxs.index(idx) for idx in self.loops if idx in idxs]

                tns = self(tns)  # evaluate the tensor

                # if there are fewer indicies than dimensions, add the remaining
                # dimensions as if they werent permutated
                if hasattr(tns, "ndim") and len(perm) < tns.ndim:
                    perm += list(range(len(perm), tns.ndim))

                tns = xp.permute_dims(tns, perm)  # permute the dimensions
                return xp.expand_dims(
                    tns,
                    [i for i in range(len(self.loops)) if self.loops[i] not in idxs],
                )

            # access a tensor with only one indirect access index
            case ein.Access(tns, idxs) if len(idxs) == 1:
                idx = self(idxs[0])
                tns = self(tns)  # evaluate the tensor

                flat_idx = (
                    idx if idx.ndim == 1 else xp.ravel_multi_index(idx.T, tns.shape)
                )
                return tns.flat[flat_idx]  # return a 1-d array by definition

            # access a tensor with a mixture of indices and other expressions
            case ein.Access(tns, idxs):
                assert self.loops is not None
                true_idxs = node.get_idxs()  # true field iteratior indicies
                assert all(isinstance(idx, ein.Index) for idx in true_idxs)

                # evaluate the tensor to access
                tns = self(tns)
                assert len(idxs) == len(tns.shape)

                # Classify indices and determine their parent groups
                idx_sizes = []  # Track size of each index dimension

                # For each index, determine which true_idx it depends on
                parent_indices = []
                for i, idx in enumerate(idxs):
                    if isinstance(idx, ein.Index):
                        parent_indices.append(idx)
                        idx_sizes.append(tns.shape[i])
                    else:
                        child_idxs = idx.get_idxs()
                        parent_indices.append(
                            list(child_idxs)[0] if len(child_idxs) == 1 else idx
                        )
                        idx_sizes.append(None)  # Will be determined later

                # Find unique parent groups and their positions
                # Create a mapping from parent to group index
                unique_parents = []
                parent_to_group = {}
                for parent in parent_indices:
                    found = False
                    for i, up in enumerate(unique_parents):
                        # For ein.Index, compare by name; for others by identity
                        if isinstance(parent, ein.Index) and isinstance(up, ein.Index):
                            if parent.name == up.name:
                                parent_to_group[id(parent)] = i
                                found = True
                                break
                        elif parent is up:
                            parent_to_group[id(parent)] = i
                            found = True
                            break
                    if not found:
                        parent_to_group[id(parent)] = len(unique_parents)
                        unique_parents.append(parent)

                # Build group positions from the mapping
                group_positions = [[] for _ in range(len(unique_parents))]
                for i, parent in enumerate(parent_indices):
                    group_idx = parent_to_group[id(parent)]
                    group_positions[group_idx].append(i)

                # Build ranges for cartesian product
                group_ranges = []
                for positions in group_positions:
                    first_pos = positions[0]
                    if isinstance(idxs[first_pos], ein.Index):
                        group_ranges.append(xp.arange(tns.shape[first_pos]))
                    else:
                        indirect_vals = self(idxs[first_pos]).flatten()
                        group_ranges.append(xp.arange(len(indirect_vals)))
                        # Update sizes for all positions in this group
                        for p in positions:
                            idx_sizes[p] = len(indirect_vals)

                # Compute cartesian product
                group_grids = xp.meshgrid(*group_ranges, indexing="ij")
                group_combos = xp.stack([g.flatten() for g in group_grids], axis=-1)

                # Build index combinations
                combo_idxs = xp.empty(
                    (group_combos.shape[0], len(idxs)), dtype=xp.int64
                )

                # Pre-evaluate all indirect indices
                indirect_vals_cache = {}
                for i, idx in enumerate(idxs):
                    if not isinstance(idx, ein.Index):
                        indirect_vals_cache[i] = self(idx).flatten()

                # Fill combo_idxs using vectorized operations where possible
                for group_idx, positions in enumerate(group_positions):
                    group_iterations = group_combos[:, group_idx]
                    for pos in positions:
                        if isinstance(idxs[pos], ein.Index):
                            combo_idxs[:, pos] = group_iterations
                        else:
                            combo_idxs[:, pos] = indirect_vals_cache[pos][
                                group_iterations
                            ]

                # evaluate the output tensor as a flat array
                flat_idx = xp.ravel_multi_index(combo_idxs.T, tns.shape)
                tns = xp.take(tns, flat_idx)

                # calculate child idxs, idxs computed using the parent "true idxs"
                child_idxs = {
                    parent_idx: [
                        child_idx
                        for child_idx in idxs
                        if (parent_idx in child_idx.get_idxs())
                    ]
                    for parent_idx in true_idxs
                }

                # we assert that all the indirect access indicies
                # from the parent idxs have the same size
                child_idxs_size = {
                    parent_idx: [
                        idx_sizes[idxs.index(child_idx)]
                        for child_idx in child_idxs[parent_idx]
                    ]
                    for parent_idx in true_idxs
                }
                assert all(
                    child_idxs_size[parent_idx].count(child_idxs_size[parent_idx][0])
                    == len(child_idxs_size[parent_idx])
                    for parent_idx in true_idxs
                )

                # a mapping from each idx to its axis wrt to current shape
                idxs_axis = {idx: i for i, idx in enumerate(idxs)}

                true_idxs = list(true_idxs)
                true_idxs = sorted(
                    true_idxs, key=lambda idx: idxs_axis[child_idxs[idx][0]]
                )

                # calculate the final shape of the tensor
                # we merge the child idxs to get the final shape
                # that matches the true idxs
                final_shape = tuple(
                    idx_sizes[idxs.index(child_idxs[parent_idx][0])]
                    for parent_idx in true_idxs
                )
                tns = tns.reshape(final_shape)

                # permute and broadcast the tensor to be
                # compatible with rest of expression
                perm = [true_idxs.index(idx) for idx in self.loops if idx in true_idxs]
                tns = xp.permute_dims(tns, perm)
                return xp.expand_dims(
                    tns,
                    [
                        i
                        for i in range(len(self.loops))
                        if self.loops[i] not in true_idxs
                    ],
                )

            case ein.Plan(bodies):
                res = None
                for body in bodies:
                    res = self(body)
                return res
            case ein.Produces(args):
                return tuple(self(arg) for arg in args)

            # get non-zero elements/data array of a sparse tensor
            case ein.GetAttribute(obj, ein.Literal("elems"), _):
                obj = self(obj)
                assert isinstance(ftype(obj), SparseTensorFType)
                assert isinstance(obj, SparseTensor)
                return obj.data
            # get coord array of a sparse tensor
            case ein.GetAttribute(obj, ein.Literal("coords"), dim):
                obj = self(obj)
                assert isinstance(ftype(obj), SparseTensorFType)
                assert isinstance(obj, SparseTensor)

                # return the coord array for the given dimension or all dimensions
                return obj.coords if dim is None else obj.coords[:, dim]
            # gets the shape of a sparse tensor at a given dimension
            case ein.GetAttribute(obj, ein.Literal("shape"), dim):
                obj = self(obj)
                assert isinstance(ftype(obj), SparseTensorFType)
                assert isinstance(obj, SparseTensor)
                assert dim is not None

                # return the shape for the given dimension
                return obj.shape[dim]

            # standard einsum
            case ein.Einsum(op, ein.Alias(tns), idxs, arg) if all(
                isinstance(idx, ein.Index) for idx in idxs
            ):
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
                raise NotImplementedError(
                    "Indirect einsum assignment is not implemented"
                )
            case _:
                raise ValueError(f"Unknown einsum type: {type(node)}")
