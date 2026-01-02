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
        dict[ein.Alias, tuple[ein.Field, ...]],
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
    ) -> tuple[AssemblyLibrary, ein.EinsumStatement, dict[ein.Alias, TensorFType]]:
        for opt in self.opts:
            term, bindings = opt(term, bindings or {})
        return self.ctx(term, bindings)


def compute_shape_vars(
    prgm: ein.EinsumStatement,
    bindings: dict[ein.Alias, TensorFType],
) -> dict[ein.Alias, tuple[ein.Field, ...]]:
    groups = {}
    dim_bindings = {}
    for var, tns in bindings.items():
        idxs = [ein.Field(f"{var.name}_i_{i}") for i in range(tns.ndim)]
        for idx in idxs:
            groups[idx] = set(idx)
        dim_bindings[var] = tuple[idxs]

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
            case ein.Einsum(_, lhs, arg, lhs_idxs):
                idx_bindings = dict(zip(lhs_idxs, dim_bindings[lhs], strict=True))
                for node in PostOrderDFS(arg):
                    match node:
                        case ein.Access(var, idxs):
                            for idx, dim in zip(idxs, dim_bindings[var], strict=True):
                                if idx in idx_bindings:
                                    merge_dim_groups(idx_bindings[idx], dim)
                                else:
                                    idx_bindings[idx] = dim
                        case _:
                            pass
            case _:
                pass

    prgm.infer_dimmap(merge_dim_groups, dim_bindings)

    group_names = {}

    for group in groups:
        if None in group:
            group_names[group] = None
        else:
            group_names[group] = f"i_{len(group_names)}"

    return {
        var: tuple(ein.Field(group_names[groups[idx]]) for idx in idxs)
        for var, idxs in dim_bindings.items()
    }
