import pytest

import numpy as np

import finchlite
from finchlite import (
    NumpyBufferFType,
    asarray,
    dense,
    element,
    fiber_tensor,
)
from finchlite.interface.lazy import EyeTensor
from finchlite.tensor import BufferizedNDArray


def test_fiber_tensor_attributes():
    fmt = fiber_tensor(dense(dense(element(0.0, finchlite.float64))))
    shape = (3, 4)
    arr = np.ones(shape)
    a = asarray(arr, format=fmt)

    # Check shape attribute
    assert a.shape == shape

    # Check ndim
    assert a.ndim == 2

    # Check shape_type
    assert a.shape_type == (finchlite.intp, finchlite.intp)

    # Check element_type
    assert a.element_type == finchlite.float64

    # Check fill_value
    assert a.fill_value == 0

    # Check position_type
    assert a.position_type == finchlite.intp

    # Check buffer_format exists
    assert a.buffer_factory == NumpyBufferFType


def test_bufferized_ndarray_fill_value_dtype():
    for dtype in (np.int32, np.float32, np.bool_):
        arr = np.ones((2, 3), dtype=dtype)
        a = BufferizedNDArray.from_numpy(arr, fill_value=0)

        assert a.fill_value.dtype == arr.dtype
        assert a.ftype.fill_value.dtype == arr.dtype
        assert finchlite.lazy(a).fill_value.dtype == arr.dtype
        assert asarray(a, copy=True).fill_value.dtype == arr.dtype
        assert a[0:1].fill_value.dtype == arr.dtype
        assert a.reshape((3, 2)).fill_value.dtype == arr.dtype

    full = finchlite.full((2, 3), 7, dtype=np.int32)
    assert full.fill_value.dtype == np.dtype(np.int32)


def test_bufferized_ndarray_custom_fill_value():
    arr = np.ones((2, 3), dtype=np.float32)
    a = BufferizedNDArray.from_numpy(arr, fill_value=2)

    assert a.fill_value == np.float32(2)
    assert a.ftype.fill_value == np.float32(2)
    assert np.all(a.ftype.construct((2, 3)).to_numpy() == np.float32(2))

    x = BufferizedNDArray.from_numpy(arr, fill_value=np.nan)
    y = BufferizedNDArray.from_numpy(arr, fill_value=np.nan)
    assert finchlite.same(x.fill_value, y.fill_value)
    assert x.ftype == y.ftype
    assert hash(x.ftype) == hash(y.ftype)


def test_fiber_tensor():
    fmt = fiber_tensor(
        dense(
            dense(
                element(np.int64(0), finchlite.int64, finchlite.intp, NumpyBufferFType)
            )
        )
    )

    asarray(np.arange(12).reshape((3, 4)), format=fmt)


@pytest.mark.parametrize(
    "make_tensor, expected",
    [
        (lambda: finchlite.eye(3, 4, dtype=np.int32), np.eye(3, 4, dtype=np.int32)),
        (
            lambda: finchlite.eye(3, 4, k=1, dtype=np.int32),
            np.eye(3, 4, k=1, dtype=np.int32),
        ),
    ],
)
def test_matrix_pattern_tensors(make_tensor, expected):
    tensor = make_tensor()

    assert tensor.shape == expected.shape
    assert tensor.fill_value.dtype == expected.dtype
    np.testing.assert_array_equal(tensor.to_numpy(), expected)


def test_lazy_matrix_pattern_tensor_compute():
    tensor = finchlite.lazy(EyeTensor((2, 3), dtype=np.int32))
    result = finchlite.compute(tensor)

    assert tensor.shape == (2, 3)
    assert result.fill_value.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(result.to_numpy(), np.eye(2, 3, dtype=np.int32))


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_triu_tril(k):
    arr = np.arange(12, dtype=np.int32).reshape((3, 4))

    np.testing.assert_array_equal(
        finchlite.triu(arr, k=k).to_numpy(), np.triu(arr, k=k)
    )
    np.testing.assert_array_equal(
        finchlite.tril(arr, k=k).to_numpy(), np.tril(arr, k=k)
    )


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_diag(k):
    vec = np.array([1, 2, 3], dtype=np.int32)
    mat = np.arange(12, dtype=np.int32).reshape((3, 4))

    np.testing.assert_array_equal(
        finchlite.diag(vec, k=k).to_numpy(), np.diag(vec, k=k)
    )
    np.testing.assert_array_equal(
        finchlite.diag(mat, k=k).to_numpy(), np.diag(mat, k=k)
    )


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_trace(offset):
    arr = np.arange(12, dtype=np.int32).reshape((3, 4))

    assert finchlite.trace(arr, offset=offset).item() == np.trace(arr, offset=offset)


def test_lazy_array_api_matrix_functions():
    arr = np.arange(12, dtype=np.int32).reshape((3, 4))
    x = finchlite.lazy(arr)

    np.testing.assert_array_equal(
        finchlite.compute(finchlite.triu(x, k=1)).to_numpy(), np.triu(arr, k=1)
    )
    np.testing.assert_array_equal(
        finchlite.compute(finchlite.tril(x, k=-1)).to_numpy(), np.tril(arr, k=-1)
    )
    np.testing.assert_array_equal(
        finchlite.compute(finchlite.diag(x, k=1)).to_numpy(), np.diag(arr, k=1)
    )
    assert finchlite.compute(finchlite.trace(x, offset=1)).item() == np.trace(
        arr, offset=1
    )
