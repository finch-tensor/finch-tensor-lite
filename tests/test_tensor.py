import numpy as np

import finchlite
from finchlite import (
    NumpyBufferFType,
    asarray,
    dense,
    element,
    fiber_tensor,
)
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


def test_index_tensor_returns_linear_indices():
    import importlib

    lazy_interface = importlib.import_module("finchlite.interface.lazy")

    tns = lazy_interface.IndexTensor((2, 3), np.int64)

    assert tns.shape == (2, 3)
    assert tns.element_type == finchlite.int64
    assert tns.fill_value == np.int64(0)
    assert tns[0, 0].item() == np.int64(0)
    assert tns[0, 2].item() == np.int64(2)
    assert tns[1, 0].item() == np.int64(3)
    assert tns[1, 2].item() == np.int64(5)

    constructed = tns.ftype.construct((2, 3))

    assert constructed.ftype == tns.ftype
    assert constructed[1, 2].item() == np.int64(5)
    assert not hasattr(lazy_interface, "LinearIndicesTensor")
    assert not hasattr(lazy_interface, "LinearIndicesTensorFType")


def test_fiber_tensor():
    fmt = fiber_tensor(
        dense(
            dense(
                element(np.int64(0), finchlite.int64, finchlite.intp, NumpyBufferFType)
            )
        )
    )

    asarray(np.arange(12).reshape((3, 4)), format=fmt)
