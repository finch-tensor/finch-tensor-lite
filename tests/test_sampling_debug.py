import numpy as np
from collections import OrderedDict
import finchlite as fl
import finchlite.algebra.ffuncs as ffuncs
from finchlite.finch_logic import Field
from finchlite.autoschedule.tensor_stats.sampling_stats import (
    SamplingStatsFactory,_duj1,_dsj1,_duj2,_dsh,_dsh2,_dsh3,_dgood1)
from finchlite.autoschedule.galley.logical_optimizer import insert_statistics
from finchlite.finch_logic import (
    Aggregate,
    Field,
    Literal,
    MapJoin,
    Table,
)
import pytest


i,j,k,l = Field("i"),Field("j"),Field("k"),Field("l")
rng = np.random.default_rng(0)


def test_verify_sketch_computation(n=20,density=0.4,sample_prob=0.5,seed=0):

    rs = np.random.default_rng(seed)
    A = (rs.random((n,n)) < density).astype(float)
    B = (rs.random((n,n)) < density).astype(float)

    #masks 
    mask_i = (rs.random(n) < sample_prob).astype(float)
    mask_k = (rs.random(n) < sample_prob).astype(float)
    mask_j = (rs.random(n) < sample_prob).astype(float)


    #SamplingStats
    factory = SamplingStatsFactory(sample_prob=sample_prob,estimator="uj1")
    factory._masks[(i,n)] = mask_i
    factory._masks[(j,n)] = mask_j
    factory._masks[(k,n)] = mask_k
    
    s_a = factory(fl.asarray(A),(i,k))
    s_b = factory(fl.asarray(B),(k,j))
    mm = factory.mapjoin(ffuncs.mul,s_a,s_b)
    factory_sketch = factory.aggregate(ffuncs.add,0.0,(k,),mm).sketch

    #manually calculating
    pat_a = (A!=0).astype(float)
    pat_b = (B!=0).astype(float)
    triple = (pat_a[:,:,None]*mask_i[:,None,None]*mask_k[None,:,None]*pat_b[None,:,:]*mask_k[None,:,None]*mask_j[None,None,:])
    manual_sketch = triple.sum(axis=1)

    same = np.allclose(factory_sketch,manual_sketch)
    print(f"factory_sketch.sum()={factory_sketch.sum()}\n manual_sketch.sum()={manual_sketch.sum()}\n identical={same}")
    assert same
    print("Manual computation of sketch matches with factory sketch")


def test_estimators_isolated(D_true=100,N_true=1000000,q=0.125,trials=50,seed=0):
    rs = np.random.default_rng(seed)
    multiplicities = rs.geometric(p=0.5,size=D_true)
    ests = {name:[] for name in ["uj1","sj1","uj2","schlosser","sh2","sh3","good1"]}
    for _ in range(trials):
        counts = rs.binomial(multiplicities,q)#sampled entries out of available multiplicities
        d_n = float(np.sum(counts>0))#how many did we get in the sample 
        n = float(np.sum(counts))#how many contributed in the sample
        f_1 = float(np.sum(counts==1))
        nz = counts[counts>0]
        vals,cts = np.unique(nz,return_counts=True)
        frequencies = {int(v):float(c) for v,c in zip(vals,cts)}
        
        ests["uj1"].append(_duj1(d_n,f_1,q,n))
        ests["sj1"].append(_dsj1(d_n,q,N_true))
        ests["uj2"].append(_duj2(d_n,f_1,frequencies,q,n,N_true))
        ests["schlosser"].append(_dsh(d_n,f_1,frequencies,q,n))
        ests["sh2"].append(_dsh2(d_n,f_1,frequencies,q,n,N_true))
        ests["sh3"].append(_dsh3(d_n,f_1,frequencies,q,n))
        ests["good1"].append(_dgood1(d_n,frequencies,n,N_true))

    for name,vals in ests.items():
        vals = np.array(vals)
        log2_ratio = np.mean(np.log2(vals/D_true))
        print(f"{name} mean log2(estimate/true) = {log2_ratio:+.3f}")

def test_mapjoin():
    i,j,k = Field("i"), Field("j"), Field("k")
    data_a = np.zeros((100,100))
    data_b = np.zeros((100,100))

    data_a[:20,:] = 1.0
    data_b[:20,:] = 1.0

    ta = Table(Literal(fl.asarray(data_a)),(i,j))
    tb = Table(Literal(fl.asarray(data_b)),(j,k))

    cache = {}
    stats_factory = SamplingStatsFactory(sample_prob=1)
    insert_statistics(
        stats_factory=stats_factory,
        node=ta,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    insert_statistics(
        stats_factory=stats_factory,
        node=tb,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )
    node_mul = MapJoin(Literal(ffuncs.mul),(ta,tb))
    stats = insert_statistics(
        stats_factory=stats_factory,
        node=node_mul,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    true_nnz = float(np.count_nonzero(data_a[:,:,None]*data_b[None,:,:]))
    assert stats.estimate_non_fill_values() == pytest.approx(true_nnz,abs=1.0)

@pytest.mark.skip(reason="sampling estimators are heavily dependent on f1")
def test_aggregate():
    i,j,k = Field("i"), Field("j"), Field("k")
    data_a = np.zeros((100,100))
    data_b = np.zeros((100,100))

    data_a[:20,:] = 1.0
    data_b[:20,:] = 1.0

    ta = Table(Literal(fl.asarray(data_a)),(i,j))
    tb = Table(Literal(fl.asarray(data_b)),(j,k))

    cache = {}
    stats_factory = SamplingStatsFactory(sample_prob=0.25,estimator="schlosser")
    node = Aggregate(op=Literal(ffuncs.add),init=None,arg=MapJoin(Literal(ffuncs.mul),(ta,tb)),idxs=(j,))
    stats = insert_statistics(
        stats_factory=stats_factory,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache=cache,
    )

    print(stats.sketch)
    true_nnz = float(np.count_nonzero(np.sum((data_a[:,:,None]*data_b[None,:,:]),axis=1)))
    assert stats.estimate_non_fill_values() == pytest.approx(true_nnz,abs=1.0)

