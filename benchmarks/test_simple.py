"""
ASV simple benchmarks: time core array ops (matmul, elementwise add/multiply, sum) on
64x64 inputs, parameterized over different backends.

Run: ``pixi run asv run --bench simple_benchmarks``
"""

import pytest

import numpy as np

import finchlite as fl


@pytest.fixture
def integer_arrays():
    rng = np.random.default_rng(42)
    a = fl.asarray(rng.integers(0, 10, (32, 32)))
    b = fl.asarray(rng.integers(0, 10, (32, 32)))
    return a, b


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(fl.matmul, id="matmul"),
        pytest.param(fl.add, id="add"),
        pytest.param(fl.multiply, id="multiply"),
    ],
)
def test_ops_binary(integer_arrays, scheduler, benchmark, op):
    a, b = integer_arrays
    # Warmup
    op(a, b)

    # Benchmark
    benchmark(op, a, b)


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(fl.sum, id="sum"),
    ],
)
def test_ops_reduction(integer_arrays, scheduler, benchmark, op):
    a, _ = integer_arrays
    # Warmup
    op(a)

    # Benchmark
    benchmark(op, a)
