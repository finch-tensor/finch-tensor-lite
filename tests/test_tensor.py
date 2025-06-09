from finch import FiberTensorFormat, DenseLevelFormat, ElementLevelFormat
from finch import NumpyBufferFormat
import numpy as np

def test_fiber_tensor_attributes():
    fmt = FiberTensorFormat(DenseLevelFormat(ElementLevelFormat(0)))
    shape = (3,)
    a = fmt(shape)

    # Check shape attribute
    assert a.lvl.shape == shape

    # Check ndims
    assert a.ndims == 1

    # Check shape_type
    assert a.shape_type == (np.intp,)

    # Check element_type
    assert a.element_type == int

    # Check fill_value
    assert a.fill_value == 0

    # Check position_type
    assert a.position_type == np.intp

    # Check buffer_format exists
    assert a.buffer_format == NumpyBufferFormat(np.intp)