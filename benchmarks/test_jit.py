"""
JIT ASV benchmark: time ``fl.jit`` on a 5-matrix matmul chain.
Also measures standalone ``fl.jit`` compile time.

Run: ``poetry run asv run --bench jit_benchmarks``
"""

import numpy as np

import finchlite as fl
from finchlite import matmul


def mat_chain(dims: list[int]):
    rng = np.random.default_rng(42)
    return tuple(
        rng.integers(0, 10, (dims[i], dims[i + 1])) for i in range(len(dims) - 1)
    )


def _f1_jit(A, B, C, D, E):
    return matmul(A, matmul(B, matmul(C, matmul(D, E))))


def test_f1_jit(scheduler, benchmark):
    args = mat_chain([1, 20, 30, 40, 50, 60])

    benchmark(fl.jit(_f1_jit), *args)


def test_jit_creation(scheduler, benchmark):
    benchmark(fl.jit, _f1_jit)
