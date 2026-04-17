import pytest
import numpy as np
import finchlite as fl
from finchlite.autoschedule.tensor_stats.blocked_stats import BlockedStats, BlockedStatsFactory
from finchlite.autoschedule.tensor_stats.dc_stats import DCStatsFactory
from finchlite.finch_logic import Alias, Plan, Query, Table, Field, Produces, Literal, MapJoin
from finchlite.algebra import ffunc
from finchlite.autoschedule.executor import LogicExecutor
from finchlite.autoschedule.cache import LogicCacheLRU, LogicCacheFirst, LogicCacheLRU_Embeddings
from finchlite.finch_logic.interpreter import MockLogicLoader
from finchlite.autoschedule.tensor_stats import UniformStatsFactory, DenseStatsFactory


def test_logic_cache_first():
    raw_loader = MockLogicLoader()
    cache = LogicCacheFirst(raw_loader)
    executor = LogicExecutor(ctx=cache, cache=False)

    data_a = fl.asarray(np.random.rand(50, 50))
    data_b = fl.asarray(np.random.rand(50, 50))
    data_c = fl.asarray(np.random.rand(50, 50).astype(np.int64)) 
    i, j = Field("i"), Field("j")

    plan_1 = Plan((
        Query(Alias("out"), Table(Literal(data_a), (i, j))), 
        Produces((Table(Alias("out"), (i, j)),))
    ))

    #Initiating the cache with a prgm binding pair - Expected MISS
    plan_2 = Plan((
        Query(Alias("mult"), MapJoin(Literal(ffunc.mul), (
            Table(Literal(data_a), (i, j)), 
            Table(Literal(data_a), (i, j))
        ))),
        Produces((Table(Alias("mult"), (i, j)),))
    ))
    executor(plan_1) 

    #Case 1 : Same (prgm,binding) pair - Expected HIT
    # We should be able to fetch the kernel from the cache -> Len of cache remains same since same prgm, binding pair
    sim_plan_1 = Plan((
        Query(Alias("out"), Table(Literal(data_b), (i, j))), 
        Produces((Table(Alias("out"), (i, j)),))
    ))
    executor(sim_plan_1)
    
    # Case 2 : Diff (prgm, binding) pair  - Expected MISS
    # Passing a different program so new addition to the cache
    executor(plan_2)

    # Case 3 : Diff (prgm, binding) pair - Expected MISS
    # Passing diff bindings - Data type for c is changed
    plan_1_diff = Plan((
        Query(Alias("out"), Table(Literal(data_c), (i, j))), 
        Produces((Table(Alias("out"), (i, j)),))
    ))
    executor(plan_1_diff)


def test_logic_cache_lru():
    raw_loader = MockLogicLoader()
    cache = LogicCacheLRU(raw_loader, max_depth=2)
    
    #Creating two diff executors based on stats_factory passed
    executor_u = LogicExecutor(ctx=cache, stats_factory=UniformStatsFactory(),cache=False)
    executor_d = LogicExecutor(ctx=cache, stats_factory=DenseStatsFactory(),cache=False)
    
    i, j = Field("i"), Field("j")

    data_1 = fl.asarray(np.ones((10, 10))) 
    data_2 = fl.asarray(np.eye(10))     
    data_3 = fl.asarray(np.zeros((10, 10)))
    data_3.to_numpy()[0, 0] = 1.0         

    #Initiating the cache with a (prgm, bindngs) pair - Expected MISS
    plan_1_u = Plan((
        Query(Alias("out"), Table(Literal(data_1), (i, j))),
        Produces((Table(Alias("out"), (i, j)),))
    ))
    #Using uniform stats
    executor_u(plan_1_u)

    #Case 1 : Same prgm, bindings, statsfactory -  Expected HIT
    #Passing in the same (prgm, bindings) hence same stats to see if we get a HIT
    plan_1_u_sim = Plan((
        Query(Alias("out"), Table(Literal(data_1), (i, j))),
        Produces((Table(Alias("out"), (i, j)),))
    ))
    #Same stats as above
    executor_u(plan_1_u_sim)

    #Case 2 : Same prgm, bindings, but diff statsfactory [DenseStatsFactory] here  - Expected MISS
    plan_1_d = Plan((
        Query(Alias("out"), Table(Literal(data_1), (i, j))),
        Produces((Table(Alias("out"), (i, j)),))
    ))
    executor_d(plan_1_d)

    #Case 3 : Exceeding max depth  - Expected MISS ?
    # Cache currently has 2 entries. Adding a 3rd should pop the LRU.
    plan_2 = Plan((
        Query(Alias("out"), Table(Literal(data_2), (i, j))),
        Produces((Table(Alias("out"), (i, j)),))
    ))
    executor_u(plan_2)
    
    #Size remains unchanged as we popped one and added one
    plan_3 = Plan((
        Query(Alias("out"), Table(Literal(data_3), (i, j))),
        Produces((Table(Alias("out"), (i, j)),))
    ))
    executor_u(plan_3)


def test_logic_cache_embeddings():
    raw_loader = MockLogicLoader()
    cache = LogicCacheLRU_Embeddings(raw_loader, max_depth=3)

    executor_u = LogicExecutor(ctx=cache, stats_factory=UniformStatsFactory(),cache=False)  
    executor_d = LogicExecutor(ctx=cache, stats_factory=DenseStatsFactory(),cache=False)  
    executor_dc = LogicExecutor(ctx=cache, stats_factory=DCStatsFactory(),cache=False)


    i, j = Field("i"), Field("j")
    blocks_per_dim = {i: 2, j: 2}
    
    data_1 = fl.asarray(np.ones((10, 10))) 
    data_2 = fl.asarray(np.eye(10))     
    data_3 = fl.asarray(np.zeros((10, 10)))
    data_3.to_numpy()[0, 0] = 1.0

    # ------------------------ Simple plan --------------------------------- 


    #Initiating the cache with a (prgm, bindngs, statsfactory) pair - Expected MISS
    plan_1_u = Plan((
        Query(Alias("out"), Table(Literal(data_1), (i, j))),
        Produces((Table(Alias("out"), (i, j)),))
    ))
    #Using dense stats
    executor_d(plan_1_u)

    #Case 1 : Same prgm, bindings, statsfactory -  Expected HIT
    #Passing in the same (prgm, bindings, statsfactory) hence same stats to see if we get a HIT
    plan_1_u_sim = Plan((
        Query(Alias("out"), Table(Literal(data_2), (i, j))),
        Produces((Table(Alias("out"), (i, j)),))
    ))
    #Same stats as above
    executor_d(plan_1_u_sim)
    
    # ------------------------ Testing plans with diff bindings  ---------------------------------  
    mul_node = MapJoin(
        Literal(ffunc.mul), 
        (Table(Literal(data_1), (i, j)), Table(Literal(data_2), (i, j)))
    )
    plan_mul = Plan((
        Query(Alias("result"), mul_node),
        Produces((Table(Alias("result"), (i, j)),))
    ))

    mul_node_2 = MapJoin(
        Literal(ffunc.mul), 
        (Table(Literal(data_2), (i, j)), Table(Literal(data_3), (i, j)))
    )
    plan_mul_2 = Plan((
        Query(Alias("result"), mul_node_2),
        Produces((Table(Alias("result"), (i, j)),))
    ))

    #Case 1 : Same (prgm, bindings_ftype, statsfactory) - DenseStats so should HIT in the second go
    executor_d(plan_mul) #Expected MISS
    executor_d(plan_mul_2)
    
    #Case 2 : Same prgm, bindings_ftype, statsfactory but different stats so different embeddings - UniformStats and DCStats
    executor_u(plan_mul) #MISS
    executor_u(plan_mul_2)  #MISS

    executor_dc(plan_mul)   #MISS
    executor_dc(plan_mul_2) #MISS



