import numpy as np

from finchlite import (
    NumpyBufferFType,
    asarray,
    dense,
    element,
    fiber_tensor,
)


def test_fiber_tensor_attributes():
    fmt = fiber_tensor(dense(dense(element(0.0))))
    shape = (3, 4)
    arr = np.ones(shape)
    a = asarray(arr, format=fmt)

    # Check shape attribute
    assert a.shape == shape

    # Check ndim
    assert a.ndim == 2

    # Check shape_type
    assert a.shape_type == (np.intp, np.intp)

    # Check element_type
    assert a.element_type == np.float64

    # Check fill_value
    assert a.fill_value == 0

    # Check position_type
    assert a.position_type == np.intp

    # Check buffer_format exists
    assert a.buffer_factory == NumpyBufferFType


def test_fiber_tensor():
    fmt = fiber_tensor(
        dense(dense(element(np.int64(0), np.int64, np.intp, NumpyBufferFType)))
    )

    asarray(np.arange(12).reshape((3, 4)), format=fmt)
