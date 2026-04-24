import logging

import numpy as np

import finchlite as fl
from finchlite.algebra import ffuncs
from finchlite.autoschedule.cache import (
    LogicCacheLRU_Embeddings_Norms,
)
from finchlite.autoschedule.executor import LogicExecutor
from finchlite.autoschedule.tensor_stats import DenseStatsFactory, UniformStatsFactory
from finchlite.autoschedule.tensor_stats.blocked_stats import BlockedStatsFactory
from finchlite.autoschedule.tensor_stats.dc_stats import DCStatsFactory
from finchlite.finch_logic import (
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Table,
)
from finchlite.finch_logic.interpreter import MockLogicLoader
from finchlite.util.logging import LOG_LOGIC_POST_OPT

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)



def test_logic_cache_embeddings_norms_linf():
    logger.debug("------------- Using linf norm  ----------------")
    raw_loader = MockLogicLoader()
    cache_linf = LogicCacheLRU_Embeddings_Norms(
        raw_loader, max_depth=3, threshold=1, norm_order=np.inf
    )

    # Testing each stats with each norm
    uniform_executor_linf = LogicExecutor(
        ctx=cache_linf, stats_factory=UniformStatsFactory(), cache=False
    )
    dense_executor_linf = LogicExecutor(
        ctx=cache_linf, stats_factory=DenseStatsFactory(), cache=False
    )
    dc_executor_linf = LogicExecutor(
        ctx=cache_linf, stats_factory=DCStatsFactory(), cache=False
    )

    i, j = Field("i"), Field("j")

    data_1 = fl.asarray(np.ones((10, 10)))
    data_2 = fl.asarray(np.eye(10))
    data_3 = fl.asarray(np.zeros((10, 10)))
    data_3.to_numpy()[0, 0] = 1.0

    # ------------------------ Simple plan ---------------------------------

    # Initiating the cache with a (prgm, bindngs, statsfactory) pair
    # - Expected MISS
    plan_1_u = Plan(
        (
            Query(Alias("out"), Table(Literal(data_1), (i, j))),
            Produces((Table(Alias("out"), (i, j)),)),
        )
    )
    # Using dense stats
    dense_executor_linf(plan_1_u)

    # Case 1 : Same prgm, bindings, statsfactory -  Expected HIT
    # Passing in the same (prgm, bindings, statsfactory)
    # hence same stats to see if we get a HIT
    plan_1_u_sim = Plan(
        (
            Query(Alias("out"), Table(Literal(data_2), (i, j))),
            Produces((Table(Alias("out"), (i, j)),)),
        )
    )
    # Same stats as above
    dense_executor_linf(plan_1_u_sim)

    # ------------------- Testing plans with diff bindings  --------------------
    mul_node = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(data_1), (i, j)), Table(Literal(data_2), (i, j))),
    )
    plan_mul = Plan(
        (Query(Alias("result"), mul_node), Produces((Table(Alias("result"), (i, j)),)))
    )

    mul_node_2 = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(data_2), (i, j)), Table(Literal(data_3), (i, j))),
    )
    plan_mul_2 = Plan(
        (
            Query(Alias("result"), mul_node_2),
            Produces((Table(Alias("result"), (i, j)),)),
        )
    )

    # Case 1 : Same (prgm, bindings_ftype, statsfactory) -
    # DenseStats so should HIT in the second go
    dense_executor_linf(plan_mul)  # Expected MISS
    dense_executor_linf(plan_mul_2)

    # Case 2 : Same prgm, bindings_ftype, statsfactory but different stats
    # so different embeddings - UniformStats and DCStats
    uniform_executor_linf(plan_mul)  # MISS
    uniform_executor_linf(plan_mul_2)  # MISS

    dc_executor_linf(plan_mul)  # MISS
    dc_executor_linf(plan_mul_2)  # MISS


def test_logic_cache_embeddings_norms_l1():
    logger.debug("------------- Using l1 norm ----------------")
    raw_loader = MockLogicLoader()
    cache_l1 = LogicCacheLRU_Embeddings_Norms(
        raw_loader, max_depth=3, threshold=1, norm_order=1
    )

    # Testing each stats with each norm
    uniform_executor_l1 = LogicExecutor(
        ctx=cache_l1, stats_factory=UniformStatsFactory(), cache=False
    )
    dense_executor_l1 = LogicExecutor(
        ctx=cache_l1, stats_factory=DenseStatsFactory(), cache=False
    )
    dc_executor_l1 = LogicExecutor(
        ctx=cache_l1, stats_factory=DCStatsFactory(), cache=False
    )

    i, j = Field("i"), Field("j")

    data_1 = fl.asarray(np.ones((10, 10)))
    data_2 = fl.asarray(np.eye(10))
    data_3 = fl.asarray(np.zeros((10, 10)))
    data_3.to_numpy()[0, 0] = 1.0

    # --------------- Simple plan -------------------------

    # Initiating the cache with a (prgm, bindngs, statsfactory) pair
    # - Expected MISS
    plan_1_d = Plan(
        (
            Query(Alias("out"), Table(Literal(data_1), (i, j))),
            Produces((Table(Alias("out"), (i, j)),)),
        )
    )
    # Using dense stats
    dense_executor_l1(plan_1_d)

    # Case 1 : Same prgm, bindings, statsfactory -  Expected HIT
    # Passing in the same (prgm, bindings, statsfactory)
    # hence same stats to see if we get a HIT
    plan_1_d_sim = Plan(
        (
            Query(Alias("out"), Table(Literal(data_2), (i, j))),
            Produces((Table(Alias("out"), (i, j)),)),
        )
    )
    # Same stats as above
    dense_executor_l1(plan_1_d_sim)

    # ------------- Testing plans with diff bindings  --------------------
    mul_node = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(data_1), (i, j)), Table(Literal(data_2), (i, j))),
    )
    plan_mul = Plan(
        (Query(Alias("result"), mul_node), Produces((Table(Alias("result"), (i, j)),)))
    )

    mul_node_2 = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(data_2), (i, j)), Table(Literal(data_3), (i, j))),
    )
    plan_mul_2 = Plan(
        (
            Query(Alias("result"), mul_node_2),
            Produces((Table(Alias("result"), (i, j)),)),
        )
    )

    # Case 1 : Same (prgm, bindings_ftype, statsfactory) -
    # DenseStats so should HIT in the second go
    dense_executor_l1(plan_mul)  # Expected MISS
    dense_executor_l1(plan_mul_2)

    # Case 2 : Same prgm, bindings_ftype, statsfactory but different stats
    # so different embeddings - UniformStats and DCStats
    uniform_executor_l1(plan_mul)  # MISS
    uniform_executor_l1(plan_mul_2)  # MISS

    dc_executor_l1(plan_mul)  # MISS
    dc_executor_l1(plan_mul_2)  # MISS


def test_logic_cache_embeddings_norms_l2():
    logger.debug("------------- Using l2 norm ----------------")
    raw_loader = MockLogicLoader()
    cache_l2 = LogicCacheLRU_Embeddings_Norms(
        raw_loader, max_depth=3, threshold=1, norm_order=1
    )

    # Testing each stats with each norm
    uniform_executor_l2 = LogicExecutor(
        ctx=cache_l2, stats_factory=UniformStatsFactory(), cache=False
    )
    dense_executor_2 = LogicExecutor(
        ctx=cache_l2, stats_factory=DenseStatsFactory(), cache=False
    )
    dc_executor_l2 = LogicExecutor(
        ctx=cache_l2, stats_factory=DCStatsFactory(), cache=False
    )

    i, j = Field("i"), Field("j")

    data_1 = fl.asarray(np.ones((10, 10)))
    data_2 = fl.asarray(np.eye(10))
    data_3 = fl.asarray(np.zeros((10, 10)))
    data_3.to_numpy()[0, 0] = 1.0

    # ---------- Simple plan ---------------

    # Initiating the cache with a (prgm, bindngs, statsfactory) pair -
    # Expected MISS
    plan_1_d = Plan(
        (
            Query(Alias("out"), Table(Literal(data_1), (i, j))),
            Produces((Table(Alias("out"), (i, j)),)),
        )
    )
    # Using dense stats
    dense_executor_2(plan_1_d)

    # Case 1 : Same prgm, bindings, statsfactory -
    # Expected HIT
    # Passing in the same (prgm, bindings, statsfactory)
    # hence same stats to see if we get a HIT
    plan_1_d_sim = Plan(
        (
            Query(Alias("out"), Table(Literal(data_2), (i, j))),
            Produces((Table(Alias("out"), (i, j)),)),
        )
    )
    # Same stats as above
    dense_executor_2(plan_1_d_sim)

    # -------------- Testing plans with diff bindings  -------------
    mul_node = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(data_1), (i, j)), Table(Literal(data_2), (i, j))),
    )
    plan_mul = Plan(
        (Query(Alias("result"), mul_node), Produces((Table(Alias("result"), (i, j)),)))
    )

    mul_node_2 = MapJoin(
        Literal(ffuncs.mul),
        (Table(Literal(data_2), (i, j)), Table(Literal(data_3), (i, j))),
    )
    plan_mul_2 = Plan(
        (
            Query(Alias("result"), mul_node_2),
            Produces((Table(Alias("result"), (i, j)),)),
        )
    )

    # Case 1 : Same (prgm, bindings_ftype, statsfactory) -
    # DenseStats so should HIT in the second go
    dense_executor_2(plan_mul)  # Expected MISS
    dense_executor_2(plan_mul_2)

    # Case 2 : Same prgm, bindings_ftype, statsfactory but different stats
    # so different embeddings - UniformStats and DCStats
    uniform_executor_l2(plan_mul)  # MISS
    uniform_executor_l2(plan_mul_2)  # MISS

    dc_executor_l2(plan_mul)  # MISS
    dc_executor_l2(plan_mul_2)  # MISS


def test_blocked_vector_embedding():
    logger.debug("------------- Using blocked stats for testing ----------------")
    raw_loader = MockLogicLoader()
    cache_linf = LogicCacheLRU_Embeddings_Norms(
        raw_loader, max_depth=3, threshold=1, norm_order=1
    )

    i, j = Field("i"), Field("j")

    blocked_executor_linf = LogicExecutor(
        ctx=cache_linf,
        stats_factory=BlockedStatsFactory(DenseStatsFactory()),
        cache=False,
    )

    data_1 = fl.asarray(np.ones((10, 10)))

    plan_1_d = Plan(
        (
            Query(Alias("out"), Table(Literal(data_1), (i, j))),
            Produces((Table(Alias("out"), (i, j)),)),
        )
    )

    blocked_executor_linf(plan_1_d)
    blocked_executor_linf(plan_1_d)
