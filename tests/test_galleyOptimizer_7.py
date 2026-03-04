"""
TEST 7
Nested aggregates: out = sum_i sum_j (A[i,j] * B[j,k])

Exercises the structure where:
  - idx_starting_root[j] = MapJoin (inner arg of inner Aggregate)
  - idx_starting_root[i] = inner Aggregate (outer arg of outer Aggregate)

Correctness: Verified against NumPy.
"""
import numpy as np
import finchlite as fl

# 2x2 matrices - same as test 5
A = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
B = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

# out = sum_i sum_j (A[i,j] * B[j,k]) = sum over both axes
# Equivalent to sum(A @ B) or np.sum(np.array(A) @ np.array(B))
out = fl.compute(
    fl.sum(fl.lazy(A) @ fl.lazy(B)),  # no axis = sum over all
    ctx=fl.INTERPRET_NOTATION_GALLEY,
)

expected = np.sum(np.array(A) @ np.array(B))
assert np.allclose(out, expected), f"Got {out}, expected {expected}"

print("out = sum_i sum_j (A[i,j] * B[j,k]):", out)