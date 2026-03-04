"""
TEST 2

Test Galley on its own (no pipieline) with a * b and triangle summation.
"""
# Run finch and test GalleyLogicalOptimizer with 2 arrays.
import operator
import numpy as np
import finchlite as fl
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


def test_galley_logical_optimizer():
    """Test GalleyLogicalOptimizer with 2 arrays: out = a * b (elementwise)."""
    from finchlite.autoschedule.GalleyLogicalOptimizer import (
        GalleyLogicalOptimizer,
        optimize_plan,
    )
    from finchlite.galley.TensorStats import DenseStats
    a = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = Alias("out")
    plan = Plan(
        (
            Query(
                out,
                MapJoin(
                    Literal(operator.mul),
                    (
                        Table(Literal(a), (Field("i"), Field("j"))),
                        Table(Literal(b), (Field("i"), Field("j"))),
                    ),
                ),
            ),
            Produces((out,)),
        )
    )

    optimizer = GalleyLogicalOptimizer(DenseStats, ctx=None)
    result = optimizer(plan, bindings={})

    assert isinstance(result, Plan)
    assert len(result.bodies) >= 1
    assert isinstance(result.bodies[-1], Produces)
    for body in result.bodies[:-1]:
        assert isinstance(body, Query)

    result2 = optimize_plan(plan, DenseStats, {})
    assert isinstance(result2, Plan)
    assert isinstance(result2.bodies[-1], Produces)

    print("test_galley_logical_optimizer (2 arrays): OK")
    print("result (optimizer):", result)
    print("result bodies:", result.bodies)
    print("result2 (optimize_plan):", result2)
    print("result2 bodies:", result2.bodies)
    
def test_greedy_example():
    """Example: out = sum_i sum_j (A[i,j] * B[j,k]) via GalleyLogicalOptimizer."""
    import operator

    from finchlite.autoschedule.GalleyLogicalOptimizer import GalleyLogicalOptimizer
    from finchlite.finch_logic import (
        Aggregate,
        Alias,
        Field,
        Literal,
        MapJoin,
        Plan,
        Produces,
        Query,
        Table,
    )
    from finchlite.galley.TensorStats import DenseStats

    A = fl.asarray(np.ones((2, 3)))
    B = fl.asarray(np.ones((3, 4)))
    out = Alias("out")

    # out = sum_i sum_j (A[i,j] * B[j,k])
    plan = Plan(
        (
            Query(
                out,
                Aggregate(
                    Literal(operator.add),
                    Literal(0.0),
                    Aggregate(
                        Literal(operator.add),
                        Literal(0.0),
                        MapJoin(
                            Literal(operator.mul),
                            (
                                Table(Literal(A), (Field("i"), Field("j"))),
                                Table(Literal(B), (Field("j"), Field("k"))),
                            ),
                        ),
                        (Field("i"),),
                    ),
                    (Field("j"),),
                ),
            ),
            Produces((out,)),
        )
    )

    optimizer = GalleyLogicalOptimizer(DenseStats, ctx=None)
    result = optimizer(plan, bindings={})

    print("Greedy example (GalleyLogicalOptimizer): out = sum_i sum_j (A[i,j] * B[j,k])")
    print("Number of bodies:", len(result.bodies))
    for i, body in enumerate(result.bodies):
        if isinstance(body, Query):
            print(f"  Query {i+1}: {body.lhs.name} = {body.rhs}")
        else:
            print(f"  {i+1}: Produces({', '.join(a.name for a in body.args)})")
    print("test_greedy_example: OK")


test_greedy_example()

"""
Output: Runs correctly
"""