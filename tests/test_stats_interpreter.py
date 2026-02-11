from operator import add, mul

import pytest

import numpy as np

import finchlite as fl
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
from finchlite.galley.TensorStats import DCStats
from finchlite.galley.TensorStats.stats_interpreter import (
    StatsInterpreter,
    calculate_estimated_error,
)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((2, 2), (2, 2)),
        ((2, 3), (3, 4)),
    ],
)
def test_stats_matrix_multiplication(shape_a, shape_b):
    a = fl.asarray(np.ones(shape_a))
    b = fl.asarray(np.ones(shape_b))

    i = Field("i")
    j = Field("j")
    k = Field("k")

    p = Plan(
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

    interpreter = StatsInterpreter(StatsImpl=DCStats)
    result_stats = interpreter(p)[0]

    expected_rows = shape_a[0]
    expected_cols = shape_b[1]

    assert result_stats.dim_sizes["i"] == expected_rows
    assert result_stats.dim_sizes["j"] == expected_cols
    assert result_stats.index_order == ("i", "j")


def test_stats_matmul_error():
    a_val = fl.asarray(np.ones((20, 30)))
    b_val = fl.asarray(np.ones((30, 20)))

    i = Field("i")
    j = Field("j")
    k = Field("k")

    p = Plan(
        (
            Query(Alias("A"), Table(Literal(a_val), (i, k))),
            Query(Alias("B"), Table(Literal(b_val), (k, j))),
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

    errors = calculate_estimated_error(node=p, StatsImpl=DCStats)

    assert errors[0] == 0.0
