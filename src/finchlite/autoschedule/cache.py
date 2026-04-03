from collections import OrderedDict
from typing import Any

from finchlite.algebra.tensor import TensorFType
from finchlite.finch_logic import (
    Alias,
    LogicLoader,
    LogicStatement,
    StatsFactory,
    TensorStats,
)


class LogicCacheFirst(LogicLoader):
    def __init__(self, ctx: LogicLoader):
        self.ctx = ctx
        self.cache: dict[tuple[Any, Any], Any] = {}

    def __call__(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        key = (prgm, tuple(bindings.items()))

        if key not in self.cache:
            self.cache[key] = self.ctx(prgm, bindings, stats, stats_factory)

        return self.cache[key]


class LogicCacheLRU(LogicLoader):
    def __init__(self, ctx: LogicLoader, max_depth: int = 10):
        self.ctx = ctx
        self.max_depth = max_depth
        self.cache: dict[tuple[Any, Any], Any] = {}

    def __call__(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):

        prgm_key = (prgm, tuple(bindings.items()))
        if prgm_key not in self.cache:
            self.cache[prgm_key] = OrderedDict()  # stats : result (kernel)

        kernels = self.cache[
            prgm_key
        ]  # getting all the kernels for the prgm and bindings

        if stats:
            for saved_stats, result in kernels.items():
                saved_stats_dict = dict(saved_stats)

                if len(stats) == len(saved_stats_dict) and all(
                    stats_factory.issimilar(stats[alias], saved_stats_dict[alias])
                    for alias in stats
                ):
                    # Keep the most used stats/kernel combo as MRU.
                    kernels.move_to_end(saved_stats)
                    return result

        result = self.ctx(prgm, bindings, stats, stats_factory)
        new_stats_key = tuple(stats.items()) if stats else ()

        kernels[new_stats_key] = result
        kernels.move_to_end(new_stats_key)

        if len(kernels) > self.max_depth:
            kernels.popitem(last=False)

        return result