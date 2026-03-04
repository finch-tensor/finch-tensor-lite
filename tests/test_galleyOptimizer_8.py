"""
TEST 8
Deeper nesting: out = sum((A @ B) @ C)

Expression: sum_i sum_j sum_k sum_l (A[i,j] * B[j,k] * C[k,l])

Structure:
  - (A @ B) = sum_j A[i,j]*B[j,k]  -> inner matmul
  - (A @ B) @ C = sum_k (AB)[i,k]*C[k,l]  -> outer matmul
  - sum((A@B)@C) = sum_i sum_l over the result -> full reduction

Four levels of aggregation: sum_i sum_j sum_k sum_l.

Correctness: Verified against NumPy.
"""
import numpy as np
import finchlite as fl

# 2x2 matrices
A = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
B = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
C = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

# out = sum((A @ B) @ C) = sum_i sum_j sum_k sum_l (A[i,j] * B[j,k] * C[k,l])
out = fl.compute(
    fl.sum((fl.lazy(A) @ fl.lazy(B)) @ fl.lazy(C)),
    ctx=fl.INTERPRET_NOTATION_GALLEY,
)

expected = np.sum((np.array(A) @ np.array(B)) @ np.array(C))
assert np.allclose(out, expected), f"Got {out}, expected {expected}"

print("out = sum((A @ B) @ C):", out)