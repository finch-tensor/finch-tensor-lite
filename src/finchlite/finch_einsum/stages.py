from abc import ABC, abstractmethod
from typing import Any

from finchlite.symbolic.traversal import PostOrderDFS

from ..algebra import TensorFType
from ..finch_assembly import AssemblyLibrary
from ..symbolic import Stage
from . import nodes as ein


class EinsumEvaluator(Stage):
    @abstractmethod
    def __call__(
        self,
        term: ein.EinsumNode,
        bindings: dict[ein.Alias, Any] | None = None,
    ) -> Any | tuple[Any, ...]:  # TODO eventually Any->Tensor
        """
        Evaluate the given logic.
        """


class EinsumLoader(ABC):
    @abstractmethod
    def __call__(
        self, term: ein.EinsumStatement, bindings: dict[ein.Alias, TensorFType]
    ) -> tuple[
        AssemblyLibrary,
        dict[ein.Alias, TensorFType],
        dict[ein.Alias, tuple[ein.Index, ...]],
    ]:
        """
        Generate Finch Library from the given logic and input types, with a
        single method called main which implements the logic. Also return a
        dictionary including additional tables needed to run the kernel.
        """


class EinsumTransform(ABC):
    @abstractmethod
    def __call__(
        self, term: ein.EinsumStatement, bindings: dict[ein.Alias, TensorFType]
    ) -> tuple[ein.EinsumStatement, dict[ein.Alias, TensorFType]]:
        """
        Transform the given logic term into another logic term.
        """


class OptEinsumLoader(EinsumLoader):
    def __init__(self, *opts: EinsumTransform, ctx: EinsumLoader):
        self.ctx = ctx
        self.opts = opts

    def __call__(
        self,
        term: ein.EinsumStatement,
        bindings: dict[ein.Alias, TensorFType],
    ) -> tuple[
        AssemblyLibrary,
        dict[ein.Alias, TensorFType],
        dict[ein.Alias, tuple[ein.Index, ...]],
    ]:
        for opt in self.opts:
            term, bindings = opt(term, bindings or {})
        return self.ctx(term, bindings)


def compute_shape_vars(
    prgm: ein.EinsumStatement,
    bindings: dict[ein.Alias, TensorFType],
) -> dict[ein.Alias, tuple[ein.Index, ...]]:
    groups: dict[ein.Index, set[ein.Index]] = {}
    dim_bindings: dict[ein.Alias, tuple[ein.Index, ...]] = {}
    for var, tns in bindings.items():
        idxs = [ein.Index(f"{var.name}_i_{i}") for i in range(tns.ndim)]
        for idx in idxs:
            groups[idx] = {idx}
        dim_bindings[var] = tuple(idxs)

    def merge_dim_groups(dim1, dim2):
        if groups[dim1] is groups[dim2]:
            return dim1
        if len(groups[dim1]) < len(groups[dim2]):
            dim1, dim2 = dim2, dim1
        groups[dim1].update(groups[dim2])
        for idx in groups[dim2]:
            groups[idx] = dim1
        return dim1

    for stmt in PostOrderDFS(prgm):
        match stmt:
            case ein.Einsum(_, lhs, lhs_idxs, arg):
                assert all(isinstance(idx, ein.Index) for idx in lhs_idxs)
                idx_bindings = dict(zip(lhs_idxs, dim_bindings[lhs], strict=True))
                for node in PostOrderDFS(arg):
                    match node:
                        case ein.Access(var, idxs_2):
                            for idx_2, dim in zip(
                                idxs_2, dim_bindings[var], strict=True
                            ):
                                assert isinstance(idx_2, ein.Index)
                                if idx_2 in idx_bindings:
                                    merge_dim_groups(idx_bindings[idx_2], dim)
                                else:
                                    idx_bindings[idx_2] = dim
                        case _:
                            pass
            case _:
                pass

    group_names: dict[set[ein.Index], ein.Index] = {}

    for group in groups.values():
        if group not in group_names:
            group_names[group] = ein.Index(f"i_{len(group_names)}")

    return {
        var: tuple(group_names[groups[idx]] for idx in idxs)
        for var, idxs in dim_bindings.items()
    }
