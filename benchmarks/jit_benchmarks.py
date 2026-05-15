"""
JIT ASV benchmark: time ``fl.jit`` on a 5-matrix matmul chain.
Also measures standalone ``fl.jit`` compile time.

Run: ``poetry run asv run --bench jit_benchmarks``
"""

import numpy as np

import finchlite as fl
from finchlite import matmul
from finchlite.autoschedule.default_schedulers import INTERPRET_NOTATION_GALLEY


class JITCompare:
    timeout = 120

    def mat(self, n, m):
        rng = np.random.default_rng()
        return fl.asarray(rng.integers(0, 10, (n, m)))

    def mat_chain(self, dims: list[int]):
        return [self.mat(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

    def setup(self):
        fl.set_default_scheduler(ctx=INTERPRET_NOTATION_GALLEY)
        dims = [1, 20, 30, 40, 50, 60]
        self.A, self.B, self.C, self.D, self.E = self.mat_chain(dims)

    def time_f1_jit(self):
        @fl.jit
        def _f1_jit(A, B, C, D, E):
            return matmul(A, matmul(B, matmul(C, matmul(D, E))))

        return _f1_jit(self.A, self.B, self.C, self.D, self.E)

    def time_jit_creation(self):
        def _f1(A, B, C, D, E):
            return matmul(A, matmul(B, matmul(C, matmul(D, E))))

        return fl.jit(_f1)
