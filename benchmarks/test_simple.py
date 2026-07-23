"""
ASV simple benchmarks: time core array ops (matmul, elementwise add/multiply, sum) on
64x64 inputs, parameterized over different backends.

Run: ``pixi run benchmark``
"""

import pytest

import numpy as np

import finch as ft


@pytest.fixture
def integer_arrays():
    rng = np.random.default_rng(42)
    a = ft.asarray(rng.integers(0, 10, (32, 32)))
    b = ft.asarray(rng.integers(0, 10, (32, 32)))
    return a, b


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(ft.matmul, id="matmul"),
        pytest.param(ft.add, id="add"),
        pytest.param(ft.multiply, id="multiply"),
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
        pytest.param(ft.sum, id="sum"),
    ],
)
def test_ops_reduction(integer_arrays, scheduler, benchmark, op):
    a, _ = integer_arrays
    # Warmup
    op(a)

    # Benchmark
    benchmark(op, a)
