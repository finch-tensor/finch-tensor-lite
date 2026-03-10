import pytest

import numpy as np

import finchlite as fl
from finchlite import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    NumpyBuffer,
    NumpyBufferFType,
    SparseListLevel,
    dense,
    element,
    fiber_tensor,
)


@pytest.mark.usefixtures("numba_compiler")
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_selected_ops(dtype):

    ptr = NumpyBuffer(np.array([0, 1, 3, 4], dtype=np.intp))
    idx = NumpyBuffer(np.array([0, 0, 1, 2], dtype=np.intp))
    data = NumpyBuffer(np.array([1, 1, 2, 1], dtype=dtype))

    a = FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(element(dtype(0), dtype, np.intp, NumpyBufferFType), data),
                np.intp(3),
                ptr,
                idx,
            ),
            np.intp(3),
        )
    )
    a_np = np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]], dtype=dtype)

    fmt = fiber_tensor(
        dense(dense(element(dtype(0), dtype, np.intp, NumpyBufferFType)))
    )
    b_np = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]], dtype=dtype)
    b = fl.asarray(b_np, format=fmt)

    la = fl.lazy(a)
    lb = fl.lazy(b)

    plan = fl.sum(la, axis=0)
    res = fl.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), np.sum(a_np, axis=0))

    plan = fl.sum(la, axis=1)
    res = fl.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), np.sum(a_np, axis=1))

    plan = fl.multiply(la, lb)
    res = fl.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), a_np * b_np)

    plan = fl.add(la, lb)
    res = fl.compute(plan)
    np.testing.assert_array_equal(res.to_numpy(), a_np + b_np)
