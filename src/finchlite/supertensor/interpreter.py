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

                # ========== STEP 1 ==========
                # Group the indices which appear in exactly the same set of tensors.
                input_idxs = []
                for access in accesses:
                    tns_name = access.tns.name
                    idxs = [idx.name for idx in access.idxs]
                    input_idxs.append((tns_name, idxs))

                output_idxs = [idx.name for idx in output_idxs]
                idx_groups = SuperTensorEinsumInterpreter._group_indices(output_idxs, input_idxs)

                # ========== STEP 2 ==========
                # Assign a new index name to each group of original indices.
                tensor_set_to_new_idx = {}
                for k, (tensor_set, _) in enumerate(idx_groups):
                    tensor_set_to_new_idx[tensor_set] = f"i{k}"

                # ========== STEP 3 ==========
                # Compute the corrected SuperTensor representation for each tensor.

                inputs = []
                for access in accesses:
                    tns_name = access.tns.name
                    supertensor = self.bindings[access.tns.name]
                    idxs = [idx.name for idx in access.idxs]
                    inputs.append((tns_name, supertensor, idxs))

                corrected_bindings = {}
                corrected_idx_lists = {}

                for tns_name, supertensor, input_idxs in inputs:
                    map = []
                    idx_list = []

                    for (tns_set, idx_group) in idx_groups:
                        if tns_name in tns_set:
                            new_idx = tensor_set_to_new_idx[tns_set]
                            idx_list.append(new_idx)

                            logical_modes = []
                            for idx in idx_group:
                                logical_modes.append(input_idxs.index(idx))

                            map.append(logical_modes)

                    # Restore the logical shape of the SuperTensor.
                    logical_tns = np.empty(supertensor.shape, dtype=supertensor.base.dtype)
                    for idx in np.ndindex(supertensor.shape):
                        logical_tns[idx] = supertensor[idx]

                    # Reshape the base tensor using the updated mode map.
                    corrected_supertensor = stns.SuperTensor.from_logical(logical_tns, map)

                    # Map the original tensor name to its corrected SuperTensor representation and corresponding index list.
                    corrected_bindings[tns_name] = corrected_supertensor
                    corrected_idx_lists[tns_name] = idx_list

                # ========== STEP 4 ==========
                # Construct the correct mode map for the output SuperTensor.
                # TODO: Fix the code repetition here...

                output_supertensor = self.bindings[output_name]

                output_map = []
                output_idx_list = []

                for (tns_set, idx_group) in idx_groups:
                    if output_name in tns_set:
                        new_idx = tensor_set_to_new_idx[tns_set]
                        output_idx_list.append(new_idx)

                        logical_modes = []
                        for idx in idx_group:
                            logical_modes.append(output_idxs.index(idx))

                        output_map.append(logical_modes)

                # Restore the logical shape of the SuperTensor.
                logical_output_tns = np.empty(output_supertensor.shape, dtype=output_supertensor.base.dtype)
                for idx in np.ndindex(output_supertensor.shape):
                    logical_output_tns[idx] = output_supertensor[idx]

                # Reshape the base tensor using the updated mode map.
                corrected_output_supertensor = stns.SuperTensor.from_logical(logical_output_tns, output_map)

                # Map the original tensor name to its corrected SuperTensor representation and corresponding index list.
                corrected_bindings[output_name] = corrected_output_supertensor
                corrected_idx_lists[output_name] = output_idx_list

                # ========== STEP 5 ==========
                # Use postwalk to replace each ein.Alias node with the proper SuperTensor representation (i.e., alias can stay the same, but update the indices).
                def reshape_supertensors(node):
                    match node:
                        case ein.Access(tns, idxs):
                            updated_idxs = [ein.Index(idx) for idx in corrected_idx_lists[tns.name]]
                            return ein.Access(tns, tuple(updated_idxs))
                        case ein.Einsum(op, ein.Alias(output_name), output_idxs, arg):
                            updated_output_idxs = [ein.Index(idx) for idx in corrected_idx_lists[output_name]]
                            return ein.Einsum(op, ein.Alias(output_name), tuple(updated_output_idxs), arg)
                        case _:
                            return node
                corrected_AST = sym.PostWalk(reshape_supertensors)

                # ========== STEP 6 ==========
                # Use a regular EinsumInterpreter to execute the einsum on the SuperTensors.
                ctx = ein.EinsumInterpreter(bindings=corrected_bindings)
                result_alias = ctx(corrected_AST)  
                output_base = corrected_bindings[result_alias[0]]

                self.bindings[output_name] = stns.SuperTensor(output_supertensor.shape, output_base, output_map)
                return (output_name,)

            case _:
                raise ValueError(f"Unknown einsum type: {type(node)}")

    @classmethod
    def _collect_accesses(cls, node: ein.EinsumExpr) -> List[ein.Access]:
        """
        Use a postorder traversal to collect all ein.Access nodes in an einsum AST.

        TODO: Not sure if this (plus the algo above) is general enough to handle nested einsum expressions?

        Args:
            node: ein.EinsumExpr
                The root node of the einsum AST, i.e., the arg field of an ein.Einsum node.

        Returns:
            List[ein.Access]
                List of ein.Access nodes found in the AST.
        """
    
        accesses = []

        def postorder(curr):
            if isinstance(curr, ein.Access):
                accesses.append(curr)
            if hasattr(curr, "children"):
                for c in curr.children:
                    postorder(c)
            
        postorder(node)
        return accesses
    
    @classmethod
    def _group_indices(cls, output_idxs: List[str], inputs: List[Tuple[str, List[str]]]) -> List[Tuple[FrozenSet[str], List[str]]]:
        """
        Groups indices based on the set of tensors they appear in.

        This function establishes the canonical ordering for the indices within each group and also for the groups themselves.

        Args:
            output_idxs: List[str]
                List of output indices.
            inputs: List[Tuple[str, List[str]]]
                List of input tensors, each represented by a tuple containing the tensor name and its list of indices.

        Returns:
            List[Tuple[FrozenSet[str], List[str]]]
                List of tuples, each containing a set of tensor names and the corresponding list of indices that appear in exactly those tensors.
        """

        # Associate each tensor with its index set.
        tensors = [(name, set(idxs)) for (name, idxs) in inputs]
        tensors.append(("out", set(output_idxs)))

        # Generate all non-empty subsets of the set of tensors.
        powerset = chain.from_iterable(combinations(tensors, t) for t in range(1, len(tensors) + 1))
        
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
    