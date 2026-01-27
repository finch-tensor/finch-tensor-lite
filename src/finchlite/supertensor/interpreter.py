import numpy as np

from ..finch_einsum import EinsumInterpreter
from ..finch_einsum import nodes as ein
from ..symbolic import Namespace, PostOrderDFS, PostWalk, Rewrite
from .supertensor import SuperTensor


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
            case ein.Einsum(_, output_tns, output_idxs, arg):
                # Collect all Access nodes in the einsum AST.
                accesses = [
                    node for node in PostOrderDFS(arg) if isinstance(node, ein.Access)
                ]

                # For each index, collect the set of tensors in which the index appears.
                idx_appearances: dict[ein.Index, set[ein.Alias]] = {}
                for access in accesses:
                    for idx in access.idxs:
                        idx_appearances.setdefault(idx, set()).add(access.tns)
                for idx in output_idxs:
                    idx_appearances.setdefault(idx, set()).add(output_tns)

                # Map each set of tensors to the list of indices.
                tensor_sets: dict[tuple[ein.Alias], list[ein.Index]] = {}
                for idx, tensors in idx_appearances.items():
                    tensor_sets.setdefault(
                        tuple(sorted(tensors, key=lambda t: t.name)), []
                    ).append(idx)

                # Assign a new index name to each group of original indices.
                idx_groups: dict[ein.Index, list[ein.Index]] = {}
                old_to_new_idx_map: dict[ein.Index, ein.Index] = {}

                namespace = Namespace()
                for idx_group in tensor_sets.values():
                    new_idx = ein.Index(namespace.freshen("i"))
                    idx_groups[new_idx] = idx_group
                    for old_idx in idx_group:
                        old_to_new_idx_map[old_idx] = new_idx

                # Compute the corrected SuperTensor representations.
                corrected_bindings: dict[str, np.ndarray] = {}
                corrected_idx_lists: dict[str, list[ein.Index]] = {}
                for access in accesses:
                    new_idx_list = sorted(
                        {old_to_new_idx_map[idx] for idx in access.idxs},
                        key=lambda i: i.name,
                    )
                    mode_map = [
                        [
                            access.idxs.index(idx)
                            for idx in idx_groups[new_idx]
                            if idx in access.idxs
                        ]
                        for new_idx in new_idx_list
                    ]

                    # Restore the logical shape of the SuperTensor.
                    supertensor = self.bindings[access.tns.name]
                    logical_tns = np.empty(
                        supertensor.shape, dtype=supertensor.base.dtype
                    )
                    for idx in np.ndindex(supertensor.shape):
                        logical_tns[idx] = supertensor[idx]

                    # Reshape the base tensor using the corrected mode map.
                    corrected_supertensor = SuperTensor.from_logical(
                        logical_tns, mode_map
                    )

                    # Map the tensor name to its corrected base tensor and index list.
                    corrected_bindings[access.tns.name] = corrected_supertensor.base
                    corrected_idx_lists[access.tns.name] = new_idx_list

                # Construct the corrected index list and mode map for the output.
                new_output_idx_list = sorted(
                    {old_to_new_idx_map[idx] for idx in output_idxs},
                    key=lambda i: i.name,
                )
                output_mode_map = [
                    [
                        output_idxs.index(idx)
                        for idx in idx_groups[new_idx]
                        if idx in output_idxs
                    ]
                    for new_idx in new_output_idx_list
                ]
                corrected_idx_lists[output_tns.name] = new_output_idx_list

                # Compute the logical shape of the output SuperTensor.
                output_shape = [0] * len(output_idxs)
                for idx in output_idxs:
                    # Find an input tensor which contains this logical index.
                    for access in accesses:
                        if idx in access.idxs:
                            supertensor = self.bindings[access.tns.name]
                            output_shape[output_idxs.index(idx)] = supertensor.shape[
                                access.idxs.index(idx)
                            ]
                            break
                output_shape = tuple(output_shape)

                # Rewrite the einsum AST to use the corrected indices.
                def reshape_supertensors(node):
                    match node:
                        case ein.Access(tns, _):
                            updated_idxs = corrected_idx_lists[tns.name]
                            return ein.Access(tns, tuple(updated_idxs))
                        case ein.Einsum(op, tns, _, arg):
                            updated_output_idxs = corrected_idx_lists[tns.name]
                            return ein.Einsum(op, tns, tuple(updated_output_idxs), arg)
                        case _:
                            return node

                corrected_AST = Rewrite(PostWalk(reshape_supertensors))(node)

                # Compute the output base tensor.
                ctx = EinsumInterpreter()
                result = ctx(corrected_AST, bindings=corrected_bindings)
                output_base = corrected_bindings[result[0]]

                # Wrap the output base tensor into a SuperTensor.
                self.bindings[output_tns.name] = SuperTensor(
                    output_shape, output_base, output_mode_map
                )
                return (output_tns.name,)
            case _:
                return None
