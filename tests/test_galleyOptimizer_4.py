"""
TEST 4
Running out = a * b + c * d, with Finch/Galley pipeline using the frontend
"""


import numpy as np
import finchlite as fl
a = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
b = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
c = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
d = fl.asarray(np.array([[1.0, 1.0], [1.0, 1.0]]))
out = fl.compute(fl.lazy(a) * fl.lazy(b) + fl.lazy(c) * fl.lazy(d), ctx=fl.INTERPRET_NOTATION_GALLEY)
print("out = a * b:", out)

"""
Output: KeyError unless we assign Produces to last query. (See GalleyLogicalOptimizer.py)
"""