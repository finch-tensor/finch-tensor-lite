import pytest

import numpy as np

from finch.codegen import NumpyBuffer, NumpyBufferFType
from finch.finch_assembly import (
    SafeBufferFType,
    make_safe,
)


@pytest.mark.parametrize(
    "array_data, valid_indices, invalid_indices",
    [
        ([1, 2, 3, 4], [0, 1, 2, 3, -1, -2, -3, -4], [4, -5, 10]),
        ([42], [0, -1], [1, -2, 5]),
        ([], [], [0, -1, 1]),
    ],
)
def test_safe_buffer_bounds_checking(array_data, valid_indices, invalid_indices):
    arr = np.array(array_data, dtype=np.float64)
    numpy_buf = NumpyBuffer(arr)
    safe_buf = make_safe(numpy_buf)

    # Test valid indices work
    for idx in valid_indices:
        # load check
        expected_value = numpy_buf.load(idx)
        if expected_value is not None:
            assert safe_buf.load(idx) == expected_value
        # store check
        safe_buf.store(idx, 99.0)
        assert safe_buf.load(idx) == 99.0

    # Test invalid indices raise IndexError
    for idx in invalid_indices:
        with pytest.raises(IndexError, match="out of bounds"):
            safe_buf.load(idx)

        with pytest.raises(IndexError, match="out of bounds"):
            safe_buf.store(idx, 99.0)

    # Test format
    assert safe_buf.ftype.element_type == numpy_buf.ftype.element_type
    assert safe_buf.ftype.length_type == numpy_buf.ftype.length_type
    assert safe_buf.ftype == SafeBufferFType(numpy_buf.ftype)
    assert safe_buf.ftype.underlying_format == numpy_buf.ftype


def test_safe_buffer_format_call():
    fmt = NumpyBufferFType(dtype=np.int32)
    safe_fmt = SafeBufferFType(fmt)
    assert safe_fmt(len=5, dtype=np.int32).underlying_buffer.ftype == NumpyBufferFType(
        np.int32
    )


def test_resize_safe_buffer():
    arr = np.array([1, 2, 3], dtype=np.float64)
    numpy_buf = NumpyBuffer(arr)
    safe_buf = make_safe(numpy_buf)

    # Resize to a larger size
    safe_buf.resize(5)
    assert safe_buf.length() == 5
    numpy_buf.resize(5)
    assert isinstance(safe_buf.underlying_buffer, NumpyBuffer)
    assert np.array_equal(safe_buf.underlying_buffer.arr, numpy_buf.arr)
