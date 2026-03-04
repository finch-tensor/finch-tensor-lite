"""
TEST 9

Exercises the "extra axis padding" path in logic_to_stats.

When Table(alias, idxs) has MORE logical indices than the alias's base stats
dimensions, we pad the extra axes with size 1.0 (singleton dimensions).

This happens when:
- expand_dims adds a singleton axis → Reorder(Table(A, i, j), i, j, k)
- The stats for the Reorder's result come from its child (2D)
- Downstream we see Table(alias, i, j, k) with 3 indices
- logic_to_stats pads the 3rd with 1.0

Expression: sum over the expanded singleton axis
- A is 2x2, expand_dims(A, axis=2) → (2,2,1)
- sum(..., axis=2) → (2,2) — reduces the singleton (no-op semantically)

Correctness: Verified against NumPy.
"""
import numpy as np
import finchlite as fl

A = fl.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))

# expand_dims adds singleton; sum over it → exercises padding (more indices than base dims)
expanded = fl.expand_dims(fl.lazy(A), axis=2)  # (2,2) -> (2,2,1)
out = fl.compute(fl.sum(expanded, axis=2), ctx=fl.INTERPRET_NOTATION_GALLEY)

expected = np.sum(np.expand_dims(np.array(A), axis=2), axis=2)
assert np.allclose(out, expected), f"Got {out}, expected {expected}"

print("out = sum(expand_dims(A, 2), axis=2):", out)
