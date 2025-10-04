import numpy as np

from finchlite import (
    DenseLevelFType,
    ElementLevelFType,
    FiberTensorFType,
    NumpyBuffer,
    NumpyBufferFType,
    dense,
    element,
    fiber_tensor,
)


def test_fiber_tensor_attributes():
    fmt = FiberTensorFType(DenseLevelFType(ElementLevelFType(0.0)))
    shape = (3,)
    a = fmt(shape=shape)

    # Check shape attribute
    assert a.shape == shape

    # Check ndim
    assert a.ndim == 1

    # Check shape_type
    assert a.shape_type == (np.intp,)

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

    fmt(shape=(3, 4), val=NumpyBuffer(np.arange(12)))
