"""
TEST 5
Running out = sum_i sum_j (A[i,j] * B[j,k]), with Finch/Galley pipeline using the frontend
"""
import numpy as np
import finchlite as fl

A = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
B = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
out = fl.compute(fl.sum(fl.lazy(A) @ fl.lazy(B), axis=0), ctx=fl.INTERPRET_NOTATION_GALLEY)
print("out = sum_i sum_j (A[i,j] * B[j,k]):", out)

"""
Output: KeyError
"""