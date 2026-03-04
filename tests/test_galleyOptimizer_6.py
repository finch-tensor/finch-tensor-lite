"""
TEST 6
More complex expression to stress the greedy optimizer.

Expression: out = sum(A @ B, axis=0) + sum(C @ D, axis=1)

In index notation:
  - Left:  sum_i sum_j (A[i,j] * B[j,k])  -> matmul then sum over rows, shape (cols_B,)
  - Right: sum_j sum_k (C[i,k] * D[k,j])  -> matmul then sum over cols, shape (rows_C,)
  - Result: elementwise add of two length-2 vectors -> (2,)

For 2x2 matrices: sum(A@B, axis=0) = [col0_sum, col1_sum], sum(C@D, axis=1) = [row0_sum, row1_sum].
Correctness: Verified against NumPy.
"""
import numpy as np
import finchlite as fl

# 2x2 matrices
A = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
B = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
C = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
D = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

# out = sum(A @ B, axis=0) + sum(C @ D, axis=1)
#       ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^
#       reduces rows          reduces cols
out = fl.compute(
    fl.sum(fl.lazy(A) @ fl.lazy(B), axis=0) + fl.sum(fl.lazy(C) @ fl.lazy(D), axis=1),
    ctx=fl.INTERPRET_NOTATION_GALLEY,
)

# NumPy reference: same expression
left = np.sum(np.array(A) @ np.array(B), axis=0)
right = np.sum(np.array(C) @ np.array(D), axis=1)
expected = left + right

assert np.allclose(out, expected), f"Got {out}, expected {expected}"

print("out = sum(A@B, axis=0) + sum(C@D, axis=1):", out)
