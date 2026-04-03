from collections import OrderedDict
from operator import add, mul

import numpy as np

import finchlite as fl
from finchlite.autoschedule.tensor_stats.dc_stats import DCStats
from finchlite.autoschedule.tensor_stats.stats_interpreter import StatsInterpreter
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)

a = fl.asarray(np.ones((3, 4)))
b = fl.asarray(np.ones((4, 2)))

i, j, k = Field("i"), Field("j"), Field("k")

plan = Plan(
    (
        Query(Alias("A"), Table(Literal(a), (i, k))),
        Query(Alias("B"), Table(Literal(b), (k, j))),
        Query(
            Alias("AB"),
            MapJoin(
                Literal(mul), (Table(Alias("A"), (i, k)), Table(Alias("B"), (k, j)))
            ),
        ),
        Query(
            Alias("C"),
            Reorder(
                Aggregate(
                    Literal(add), Literal(0), Table(Alias("AB"), (i, k, j)), (k,)
                ),
                (i, j),
            ),
        ),
        Produces((Alias("C"),)),
    )
)

interpreter = StatsInterpreter(StatsImpl=DCStats, verbose=True)
result = interpreter(plan, OrderedDict())
# stats = result[0]
# print(f"\ndims    : { {f.name: s for f, s in stats.dim_sizes.items()} }")
# print(f"fill    : {stats.fill_value}")
# print(f"est_nnz : {stats.estimate_non_fill_values()}")
