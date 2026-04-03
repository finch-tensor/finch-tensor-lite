# AI modified: 2026-04-03T01:49:31Z b3e812faf69fcf291b314f9e088ed51c02e3f98e
# AI modified: 2026-04-03T19:09:59Z 78911eec
# AI modified: 2026-04-03T19:13:17Z 78911eec
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from finchlite.algebra.tensor import Tensor, TensorFType

from ..finch_assembly import AssemblyLibrary
from ..symbolic import Stage
from . import nodes as lgc
from .tensor_stats import StatsFactory, TensorStats


class LogicEvaluator(Stage):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicNode, bindings: dict[lgc.Alias, Tensor] | None = None
    ) -> lgc.TableValue | tuple[Tensor, ...]:
        """
        Evaluate the given logic.
        """


class LogicLoader(ABC):
    @abstractmethod
    def __call__(
        self,
        term: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        dict[lgc.Alias, tuple[lgc.Field | None, ...]],
    ]:
        """
        Generate Finch Library from the given logic and input types, with a
        single method called `main` which implements the logic. Also return a
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
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        dict[lgc.Alias, tuple[lgc.Field | None, ...]],
    ]:
        for opt in self.opts:
            term, bindings = opt(term, bindings or {})
        return self.ctx(term, bindings, stats, stats_factory)


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
            groups[dim2].add(None)
            return dim2
        if dim2 is None:
            groups[dim1].add(None)
            return dim1
        if groups[dim1] is groups[dim2]:
            return dim1
        if len(groups[dim1]) < len(groups[dim2]):
            dim1, dim2 = dim2, dim1
        groups[dim1].update(groups[dim2])
        for idx in groups[dim2]:
            groups[idx] = groups[dim1]
        return dim1

    prgm.infer_dimmap(merge_dim_groups, dim_bindings)

    group_names: dict[int, lgc.Field | None] = {}

    for group in groups.values():
        if None in group:
            group_names[id(group)] = None
        elif id(group) not in group_names:
            group_names[id(group)] = lgc.Field(f"i_{len(group_names)}")

    return {
        var: tuple(
            group_names[id(groups[idx])] if idx is not None else None for idx in idxs
        )
        for var, idxs in dim_bindings.items()
    }
