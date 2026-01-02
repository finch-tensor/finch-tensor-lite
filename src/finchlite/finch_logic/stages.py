from __future__ import annotations

from abc import ABC, abstractmethod

from finchlite.algebra.tensor import Tensor, TensorFType

from ..finch_assembly import AssemblyLibrary
from ..symbolic import Stage
from . import nodes as lgc


class LogicEvaluator(Stage):
    @abstractmethod
    def __call__(
        self,
        term: lgc.LogicNode,
        bindings: dict[lgc.Alias, Tensor] | None = None,
    ) -> lgc.TableValue | tuple[Tensor, ...]:
        """
        Evaluate the given logic.
        """


class LogicLoader(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TensorFType]
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        dict[lgc.Alias, tuple[lgc.Field | None, ...]],
    ]:
        """
        Generate Finch Library from the given logic and input types, with a
        single method called main which implements the logic. Also return a
        dictionary including additional tables needed to run the kernel.
        """


class LogicTransform(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TensorFType]
    ) -> tuple[lgc.LogicStatement, dict[lgc.Alias, TensorFType]]:
        """
        Transform the given logic term into another logic term.
        """


class OptLogicLoader(LogicLoader):
    def __init__(self, *opts: LogicTransform, ctx: LogicLoader):
        self.ctx = ctx
        self.opts = opts

    def __call__(
        self,
        term: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        dict[lgc.Alias, tuple[lgc.Field | None, ...]],
    ]:
        for opt in self.opts:
            term, bindings = opt(term, bindings or {})
        return self.ctx(term, bindings)


def compute_shape_vars(
    prgm: lgc.LogicStatement,
    bindings: dict[lgc.Alias, TensorFType],
) -> dict[lgc.Alias, tuple[lgc.Field | None, ...]]:
    groups: dict[lgc.Field | None, set[lgc.Field | None]] = {}
    dim_bindings: dict[lgc.Alias, tuple[lgc.Field | None, ...]] = {}
    for var, tns in bindings.items():
        idxs = [lgc.Field(f"{var.name}_i_{i}") for i in range(tns.ndim)]
        for idx in idxs:
            groups[idx] = {idx}
        dim_bindings[var] = tuple(idxs)

    def merge_dim_groups(dim1, dim2):
        if dim1 is None:
            groups[dim2].update(None)
            return dim2
        if dim2 is None:
            groups[dim1].update(None)
            return dim1
        if groups[dim1] is groups[dim2]:
            return dim1
        if len(groups[dim1]) < len(groups[dim2]):
            dim1, dim2 = dim2, dim1
        groups[dim1].update(groups[dim2])
        for idx in groups[dim2]:
            groups[idx] = dim1
        return dim1

    prgm.infer_dimmap(merge_dim_groups, dim_bindings)

    group_names: dict[set[lgc.Field | None], lgc.Field | None] = {}

    for group in groups.values():
        if None in group:
            group_names[group] = None
        elif group not in group_names:
            group_names[group] = lgc.Field(f"i_{len(group_names)}")

    return {
        var: tuple(group_names[groups[idx]] for idx in idxs)
        for var, idxs in dim_bindings.items()
    }
