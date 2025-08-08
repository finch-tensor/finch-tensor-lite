import numpy as np

from finch import (
    DenseLevelFType,
    ElementLevelFType,
    FiberTensorFType,
    NumpyBufferFType,
)


def test_fiber_tensor_attributes():
    fmt = FiberTensorFType(DenseLevelFType(ElementLevelFType(0.0)))
    shape = (3,)
    a = fmt(shape)

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
