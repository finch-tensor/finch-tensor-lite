import logging

import numpy as np
from numpy.linalg import vector_norm

from finchlite.adv_autoschedulers.tensor_stats.numeric_stats import NumericStats
from finchlite.algebra.tensor import TensorFType
from finchlite.finch_logic import (
    Alias,
    LogicLoader,
    LogicStatement,
    StatsFactory,
    TensorStats,
)

from ..util.logging import LOG_LOGIC_POST_OPT

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)


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
            return vector_norm(dist, ord=norm_order, axis=1)

        prgm_key = (prgm, tuple(bindings.items()), stats_factory)
        if prgm_key not in self.cache:
            self.cache[prgm_key] = {
                "cached_embeddings": None,
                "kernels": [],
            }  # embeddings : result (kernel)

        entry = self.cache[prgm_key]  # fetching the cached vectors and kernels

        if stats:
            current_embedding = np.concatenate(
                [
                    s.get_embedding()
                    for s in stats.values()
                    if isinstance(s, NumericStats)
                ]
            )

            factor = vector_norm(np.ones(len(current_embedding)), ord=self.norm_order)
            current_vec = current_embedding / factor

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
