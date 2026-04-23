import logging
from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.linalg import vector_norm

from finchlite.algebra.tensor import TensorFType
from finchlite.autoschedule.tensor_stats.numeric_stats import NumericStats
from finchlite.finch_logic import (
    Alias,
    LogicLoader,
    LogicStatement,
    StatsFactory,
    TensorStats,
)

from ..util.logging import LOG_LOGIC_POST_OPT

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


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
            logger.debug("CacheFirst MISS, compiling a new kernel")
            self.cache[key] = self.ctx(prgm, bindings, stats, stats_factory)
        else:
            logger.debug("CacheFirst HIT, reusing kernel")

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
            self.cache[prgm_key] = OrderedDict()
            # (prgm, bindings) : [stats (Associated with the program) : result (kernel)]

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
                    # Keep the most used stats:kernel combo as MRU.
                    logger.debug("CacheLRU HIT, reusing kernel")
                    kernels.move_to_end(saved_stats)
                    return result

        logger.debug("CacheLRU MISS, compiling new kernel ")
        result = self.ctx(prgm, bindings, stats, stats_factory)
        new_stats_key = tuple(stats.items()) if stats else ()

        kernels[new_stats_key] = result
        kernels.move_to_end(new_stats_key)

        if len(kernels) > self.max_depth:
            kernels.popitem(last=False)

        return result


class LogicCacheLRU_Embeddings(LogicLoader):
    def __init__(self, ctx: LogicLoader, max_depth: int = 10, threshold: int = 1):
        self.ctx = ctx
        self.max_depth = max_depth
        self.cache: dict[tuple, dict] = {}
        self.threshold = threshold

    def __call__(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):

        prgm_key = (prgm, tuple(bindings.items()), stats_factory)
        if prgm_key not in self.cache:
            self.cache[prgm_key] = {
                "cached_embeddings": None,
                "kernels": [],
            }  # embeddings : result (kernel)

        entry = self.cache[prgm_key]  # fetching the cached vectors and kernels

        if stats:
            current_vec = np.concatenate(
                [
                    s.get_embedding()
                    for s in stats.values()
                    if isinstance(s, NumericStats)
                ]
            )  # concatenating the embeddings
            if entry["cached_embeddings"] is not None:
                dist = np.abs(entry["cached_embeddings"] - current_vec)
                max_dist = np.max(dist, axis=1)
                chosen_idx = np.argmin(max_dist)  # threshold = 1
                if max_dist[chosen_idx] < self.threshold:
                    logger.debug("CacheLRU_Embeddings HIT, reusing kernel")
                    return entry["kernels"][chosen_idx]

        logger.debug("CacheLRU_Embeddings MISS, compiling new kernel and embeddings")
        result = self.ctx(prgm, bindings, stats, stats_factory)

        if stats:
            if entry["cached_embeddings"] is None:
                entry["cached_embeddings"] = np.array([current_vec])
            else:
                entry["cached_embeddings"] = np.vstack(
                    [entry["cached_embeddings"], current_vec]
                )

            entry["kernels"].append(result)

            if len(entry["kernels"]) > self.max_depth:
                entry["cached_embeddings"] = np.delete(
                    entry["cached_embeddings"], 0, axis=0
                )
                entry["kernels"].pop(0)

        return result


class LogicCacheLRU_Embeddings_Norms(LogicLoader):
    def __init__(
        self,
        ctx: LogicLoader,
        max_depth: int = 10,
        threshold: int = 1,
        norm_order: float = np.inf,
    ):
        self.ctx = ctx
        self.max_depth = max_depth
        self.cache: dict[tuple, dict] = {}
        self.threshold = threshold
        self.norm_order = norm_order

    def __call__(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):

        def apply_norm(cached_matrix, current_vec, norm_order):
            dist = np.abs(cached_matrix - current_vec)
            match norm_order:
                case np.inf:
                    return vector_norm(dist, ord=np.inf, axis=1)

                case 1:
                    return vector_norm(dist, ord=1, axis=1)

                case 2:
                    return vector_norm(dist, ord=2, axis=1)

        prgm_key = (prgm, tuple(bindings.items()), stats_factory)
        if prgm_key not in self.cache:
            self.cache[prgm_key] = {
                "cached_embeddings": None,
                "kernels": [],
            }  # embeddings : result (kernel)

        entry = self.cache[prgm_key]  # fetching the cached vectors and kernels

        if stats:
            match self.norm_order:
                case np.inf:
                    current_vec = np.concatenate(
                        [
                            s.get_embedding()
                            for s in stats.values()
                            if isinstance(s, NumericStats)
                        ]
                    )
                case 1:
                    current_vec = np.concatenate(
                        [
                            (s.get_embedding() / len(s.get_embedding()))
                            for s in stats.values()
                            if isinstance(s, NumericStats)
                        ]
                    )
                case 2:
                    current_vec = np.concatenate(
                        [
                            (s.get_embedding() / np.sqrt(len(s.get_embedding())))
                            for s in stats.values()
                            if isinstance(s, NumericStats)
                        ]
                    )

            if entry["cached_embeddings"] is not None:
                distances = apply_norm(
                    entry["cached_embeddings"], current_vec, self.norm_order
                )
                chosen_idx = np.argmin(distances)  # threshold = 1
                if distances[chosen_idx] < self.threshold:
                    logger.debug("CacheLRU_Embeddings_Norms HIT, reusing kernel")
                    return entry["kernels"][chosen_idx]

        logger.debug(
            "CacheLRU_Embeddings_Norms MISS, compiling new kernel and embeddings"
        )
        result = self.ctx(prgm, bindings, stats, stats_factory)

        if stats:
            if entry["cached_embeddings"] is None:
                entry["cached_embeddings"] = np.array([current_vec])
            else:
                entry["cached_embeddings"] = np.vstack(
                    [entry["cached_embeddings"], current_vec]
                )

            entry["kernels"].append(result)

            if len(entry["kernels"]) > self.max_depth:
                entry["cached_embeddings"] = np.delete(
                    entry["cached_embeddings"], 0, axis=0
                )
                entry["kernels"].pop(0)

        return result
