import logging

import pytest

import numpy as np

import finch as ft
from finch import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    NumpyBuffer,
    NumpyBufferFType,
    SparseListLevel,
    dense,
    element,
    fiber_tensor,
    ftype,
)
from finch.symbolic.gensym import _sg

from .conftest import reset_name_counts


@pytest.mark.usefixtures("numba_compiler")
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_selected_ops(dtype):
    ptr = NumpyBuffer(np.array([0, 1, 3, 4], dtype=np.intp))
    idx = NumpyBuffer(np.array([0, 0, 1, 2], dtype=np.intp))
    data = NumpyBuffer(np.array([1, 1, 2, 1], dtype=dtype))

    elem_ftype = element(dtype(0), ftype(dtype), ftype(np.intp(0)), NumpyBufferFType)

    a = FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(elem_ftype, data),
                np.intp(3),
                ptr,
                idx,
            ),
            np.intp(3),
        )
    )
    a_np = np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]], dtype=dtype)

    fmt = fiber_tensor(
        dense(
            dense(element(dtype(0), ftype(dtype), ftype(np.intp(0)), NumpyBufferFType))
        )
    )
    b_np = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]], dtype=dtype)
    b = ft.asarray(b_np, format=fmt)

    la = ft.lazy(a)
    lb = ft.lazy(b)

    plan = ft.sum(la, axis=0)
    res = ft.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), np.sum(a_np, axis=0))

    plan = ft.sum(la, axis=1)
    res = ft.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), np.sum(a_np, axis=1))

    plan = ft.multiply(la, lb)
    res = ft.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), a_np * b_np)

    plan = ft.add(la, lb)
    res = ft.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), a_np + b_np)


def test_asm_sparse_elemwise(file_regression, caplog, numba_compiler):
    _sg.counter = 0

    dtype = np.float64
    ptr = NumpyBuffer(np.array([0, 1, 3, 4], dtype=np.intp))
    idx = NumpyBuffer(np.array([0, 0, 1, 2], dtype=np.intp))
    data = NumpyBuffer(np.array([1, 1, 2, 1], dtype=dtype))

    a = FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(
                    element(
                        dtype(0), ft.ftype(dtype), ft.ftype(np.intp), NumpyBufferFType
                    ),
                    data,
                ),
                np.intp(3),
                ptr,
                idx,
            ),
            np.intp(3),
        )
    )
    b = FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(
                    element(
                        dtype(0), ft.ftype(dtype), ft.ftype(np.intp), NumpyBufferFType
                    ),
                    data,
                ),
                np.intp(3),
                ptr,
                idx,
            ),
            np.intp(3),
        )
    )
    la = ft.lazy(a)
    lb = ft.lazy(b)

    class DummyHandler(logging.Handler):
        def __init__(self, level):
            super().__init__(level=level)
            self.records = []

        def emit(self, record):
            self.records.append(record)

    handler = DummyHandler(logging.DEBUG)
    log = logging.getLogger("finch.compile.lower")
    log.addHandler(handler)
    log.propagate = False

    with caplog.at_level(logging.DEBUG, logger="finch.compile.lower"):
        result = ft.multiply(la, lb)
        _ = ft.compute(result)

    log.propagate = True

    assert len(handler.records) == 1
    file_regression.check(
        reset_name_counts(str(handler.records[0].msg)), extension=".txt"
    )
