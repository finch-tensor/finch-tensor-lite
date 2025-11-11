import operator

import numpy as np
import copy

from ..algebra import overwrite, promote_max, promote_min
from typing import List, Tuple, Set, FrozenSet, Dict
from itertools import chain, combinations

from finchlite import finch_einsum as ein
from finchlite import symbolic as sym
from . import supertensor as stns

class SuperTensorEinsumInterpreter:
    def __init__(self, xp=None, bindings=None):
        if bindings is None:
            bindings = {}
        if xp is None:
            xp = np
        self.bindings = bindings
        self.xp = xp

    def __call__(self, node):
        match node:
            case ein.Plan(bodies):
                res = None
                for body in bodies:
                    res = self(body)
                return res
            case ein.Produces(args):
                return tuple(self(arg) for arg in args)
            case ein.Einsum(op, ein.Alias(output_name), output_idxs, arg):
                accesses = SuperTensorEinsumInterpreter._collect_accesses(arg)

                # Group the indices which appear in exactly the same set of tensors.
                output_idxs = [idx.name for idx in output_idxs]

                input_idx_list = []
                for access in accesses:
                    tns_name = access.tns.name
                    idxs = [idx.name for idx in access.idxs]
                    input_idx_list.append((tns_name, idxs))
                
                idx_groups = SuperTensorEinsumInterpreter._group_indices(output_name, output_idxs, input_idx_list)

                # Assign a new index name to each group of original indices.
                new_idxs = {}
                for k, (tensor_set, _) in enumerate(idx_groups):
                    new_idxs[tensor_set] = f"i{k}"


                # Compute the corrected SuperTensor representation for each tensor.
                inputs = []
                for access in accesses:
                    tns_name = access.tns.name
                    supertensor = self.bindings[access.tns.name]
                    idxs = [idx.name for idx in access.idxs]
                    inputs.append((tns_name, supertensor, idxs))

                corrected_bindings = {}
                corrected_idx_lists = {}
                for tns_name, supertensor, input_idx_list in inputs:
                    new_idx_list = []
                    mode_map = []
                    for (tns_set, idx_group) in idx_groups:
                        if tns_name in tns_set:
                            new_idx = new_idxs[tns_set]
                            new_idx_list.append(new_idx)

                            logical_modes = []
                            for idx in idx_group:
                                logical_modes.append(input_idx_list.index(idx))

                            mode_map.append(logical_modes)

                    # Restore the logical shape of the SuperTensor.
                    logical_tns = np.empty(supertensor.shape, dtype=supertensor.base.dtype)
                    for idx in np.ndindex(supertensor.shape):
                        logical_tns[idx] = supertensor[idx]

                    # Reshape the base tensor using the updated mode map.
                    corrected_supertensor = stns.SuperTensor.from_logical(logical_tns, mode_map)

                    # Map the tensor name to its corrected base representation and index list. 
                    corrected_bindings[tns_name] = corrected_supertensor.base
                    corrected_idx_lists[tns_name] = new_idx_list

                # Construct the correct mode map and index list for the output SuperTensor.
                # TODO: Fix the code repetition here...

                new_output_idx_list = []
                output_mode_map = []          
                for (tns_set, idx_group) in idx_groups:
                    if output_name in tns_set:
                        new_idx = new_idxs[tns_set]
                        new_output_idx_list.append(new_idx)

                        logical_modes = []
                        for idx in idx_group:
                            logical_modes.append(output_idxs.index(idx))

                        output_mode_map.append(logical_modes)

                corrected_idx_lists[output_name] = new_output_idx_list

                # Compute the shape of the output SuperTensor.
                output_shape = [0] * len(output_mode_map)
                for idx in new_output_idx_list:
                    # Find an input SuperTensor which contains this index.
                    for base_tns, idx_list in zip(corrected_bindings.values(), corrected_idx_lists.values()):
                        if idx in idx_list:
                            dim = base_tns.shape[idx_list.index(idx)]
                            output_shape[new_output_idx_list.index(idx)] = dim
                            break
                output_shape = tuple(output_shape)

                # Replace each ein.Alias node with the proper base representation (i.e., update index lists).
                def reshape_supertensors(node):
                    match node:
                        case ein.Access(tns, _):
                            # TODO: What to do when tns isn't an ein.Alias?
                            if not isinstance(tns, ein.Alias):
                                return node
                            updated_idxs = [ein.Index(idx) for idx in corrected_idx_lists[tns.name]]
                            return ein.Access(tns, tuple(updated_idxs))
                        case ein.Einsum(op, ein.Alias(output_name), _, arg):
                            updated_output_idxs = [ein.Index(idx) for idx in corrected_idx_lists[output_name]]
                            return ein.Einsum(op, ein.Alias(output_name), tuple(updated_output_idxs), arg)
                        case _:
                            return node
                        
                corrected_AST = sym.Rewrite(sym.PostWalk(reshape_supertensors))(node)

                # Use a regular EinsumInterpreter to execute the einsum on the SuperTensors.
                ctx = ein.EinsumInterpreter(bindings=corrected_bindings)
                result_alias = ctx(corrected_AST)  
                output_base = corrected_bindings[result_alias[0]]

                self.bindings[output_name] = stns.SuperTensor(output_shape, output_base, output_mode_map)
                return (output_name,)
            case _:
                pass

    @classmethod
    def _collect_accesses(cls, node: ein.EinsumExpr) -> List[ein.Access]:
        """
        Collect all `ein.Access` nodes in an einsum AST.

        Args:
            node: `ein.EinsumExpr`
                The root node of the einsum expression AST, i.e., the `arg` field of an `ein.Einsum` node.

        Returns:
            `List[ein.Access]`
                A list of `ein.Access` nodes found in the AST.
        """
    
        accesses = []

        def postorder(curr):
            match curr:
                case ein.Access():
                    for child in curr.children:
                        postorder(child)
                    accesses.append(curr)
                case _:
                    if hasattr(curr, "children"):
                        for child in curr.children:
                            postorder(child)
            
        postorder(node)
        return accesses
    
    @classmethod
    def _group_indices(cls, output_name: str, output_idxs: List[str], inputs: List[Tuple[str, List[str]]]) -> List[Tuple[FrozenSet[str], List[str]]]:
        """
        Groups indices based on the set of tensors they appear in.

        Establishes the canonical ordering for the indices within each group and also for the groups themselves.

        Args:
            output_name: `str`
                The name of output tensor.
            output_idxs: `List[str]`
                The list of indices in the output tensor.
            inputs: `List[Tuple[str, List[str]]]`
                The list of input tensors, each represented by a tuple containing the tensor name and its list of indices.

        Returns:
            `List[Tuple[FrozenSet[str], List[str]]]`
                A list of tuples, each containing a set of tensor names and the corresponding list of indices that appear in exactly those tensors.
        """

        # Associate each tensor name with its index set.
        tensors = [(name, set(idxs)) for (name, idxs) in inputs]
        tensors.append((output_name, set(output_idxs)))

        # Generate all non-empty subsets of the set of tensors.
        powerset = chain.from_iterable(combinations(tensors, n) for n in range(1, len(tensors) + 1))
        
        # Associate each subset of tensors to the group of indices that appear in exactly those tensors.
        groups = []
        for subset in powerset:
            included_tensors = [name for (name, _) in subset]
            included_idx_sets = [idxs for (_, idxs) in subset]
            excluded_idx_sets = [idxs for (name, idxs) in tensors if name not in included_tensors]
            
            included_intersection = set.intersection(*included_idx_sets)
            excluded_union = set.union(*excluded_idx_sets) if excluded_idx_sets else set()            
            idx_group = included_intersection.difference(excluded_union)

            if idx_group:
                tensor_set = frozenset(included_tensors)
                groups.append((tensor_set, sorted(list(idx_group))))
            
        return groups    