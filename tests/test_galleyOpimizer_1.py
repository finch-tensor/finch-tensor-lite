"""
TEST 1

Running out = a * b, with Finch/Galley pipeline using the frontend
"""
import numpy as np
import finchlite as fl

a = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
b = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))

out = fl.compute(fl.lazy(a) * fl.lazy(b), ctx=fl.INTERPRET_NOTATION_GALLEY)
print("out = a * b:", out)

"""
Current error: KeyError
"""
