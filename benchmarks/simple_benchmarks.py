"""
ASV simple benchmarks: time core array ops (matmul, elementwise add/multiply, sum) on
64x64 inputs, parameterized over different backends.

Run: ``poetry run asv run --bench simple_benchmarks``
"""

import numpy as np

import finchlite as fl
from finchlite.interface.fuse import COMPILE_NUMBA, INTERPRET_NOTATION_GALLEY

SCHEDULERS = {
    "interpret_galley": INTERPRET_NOTATION_GALLEY,
    "compile_numba": COMPILE_NUMBA,
}


class SimpleBenchmarks:
    params = list(SCHEDULERS.keys())
    param_names = ["scheduler"]

    def setup(self, scheduler):
        fl.set_default_scheduler(ctx=SCHEDULERS[scheduler])
        rng = np.random.default_rng(42)
        self.a = fl.asarray(rng.integers(0, 10, (64, 64)))
        self.b = fl.asarray(rng.integers(0, 10, (64, 64)))

    def time_matmul(self, scheduler):
        fl.matmul(self.a, self.b)

    def time_elementwise_add(self, scheduler):
        fl.add(self.a, self.b)

    def time_elementwise_mul(self, scheduler):
        fl.multiply(self.a, self.b)

    def time_single_summation(self, scheduler):
        fl.sum(self.a)
