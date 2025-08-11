import pytest

import numpy as np

from finch.codegen import (
    NumpyBuffer,
    NumpyBufferFType,
    SafeBufferFType,
    make_safe,
)
from finch.finch_assembly import Stack
from finch.symbolic import Context


# Note: Currently, compiler code depends on bufferftpye funcs, and they in turn depend
# on ctx. This makes it impossible to unit test buffers seperately without a Context.
class TestContext(Context):
    """
    A minimal context that allows us to test ftype functions
    """

    def __call__(self, prgm):
        return prgm

    def emit(self):
        return ""


@pytest.mark.parametrize(
    "array_data, valid_indices, invalid_indices",
    [
        ([1, 2, 3, 4], [0, 1, 2, 3, -1, -2, -3, -4], [4, -5, 10]),
        ([42], [0, -1], [1, -2, 5]),
        ([], [], [0, -1, 1]),
    ],
)
@pytest.mark.parametrize("buffer", [NumpyBuffer])
def test_safe_buffer(array_data, valid_indices, invalid_indices, buffer):
    arr = np.array(array_data, dtype=np.float64)
    buf = buffer(arr)
    safe_buf = make_safe(buf)
    safe_buf_stack = Stack(safe_buf, safe_buf.ftype)
    buf_stack = Stack(buf, buf.ftype)
    ctx = TestContext()

    # Test valid indices work
    for idx in valid_indices:
        # load check
        expected_value = buf.load(idx)
        if expected_value is not None:
            assert safe_buf.load(idx) == expected_value
        # c_load
        assert safe_buf_stack.result_format.c_load(
            ctx, safe_buf_stack, idx
        ) == buf_stack.result_format.c_load(ctx, buf_stack, idx)
        # numba_load
        assert safe_buf_stack.result_format.numba_load(
            ctx, safe_buf_stack, idx
        ) == buf_stack.result_format.numba_load(ctx, buf_stack, idx)

        # store check
        safe_buf.store(idx, 99.0)
        assert safe_buf.load(idx) == 99.0
        # c_store
        assert safe_buf_stack.result_format.c_store(
            ctx, safe_buf_stack, idx, 99.0
        ) == buf_stack.result_format.c_store(ctx, buf_stack, idx, 99.0)

        # numba_store
        assert safe_buf_stack.result_format.numba_store(
            ctx, safe_buf_stack, idx, 99.0
        ) == buf_stack.result_format.numba_store(ctx, buf_stack, idx, 99.0)

    # Test invalid indices raise IndexError
    for idx in invalid_indices:
        with pytest.raises(IndexError):
            safe_buf.load(idx)
        with pytest.raises(IndexError):
            safe_buf_stack.result_format.c_load(ctx, safe_buf_stack, idx)
        with pytest.raises(IndexError):
            safe_buf_stack.result_format.numba_load(ctx, safe_buf_stack, idx)

        with pytest.raises(IndexError):
            safe_buf.store(idx, 99.0)
        with pytest.raises(IndexError):
            safe_buf_stack.result_format.c_store(ctx, safe_buf_stack, idx, 99.0)
        with pytest.raises(IndexError):
            safe_buf_stack.result_format.numba_store(ctx, safe_buf_stack, idx, 99.0)

    # test length
    assert safe_buf.length == buf.length
    assert safe_buf.ftype.c_length(ctx, safe_buf_stack) == buf.ftype.c_length(
        ctx, buf_stack
    )
    assert safe_buf.ftype.numba_length(ctx, safe_buf_stack) == buf.ftype.numba_length(
        ctx, buf_stack
    )

    # Test format
    assert safe_buf.ftype.element_type == buf.ftype.element_type
    assert safe_buf.ftype.length_type == buf.ftype.length_type
    assert safe_buf.ftype == SafeBufferFType(buf.ftype)
    assert safe_buf.ftype.underlying_ftype == buf.ftype


def test_format_call():
    fmt = NumpyBufferFType(dtype=np.int32)
    safe_fmt = SafeBufferFType(fmt)
    assert safe_fmt(len=5, dtype=np.int32).underlying_buffer.ftype == NumpyBufferFType(
        np.int32
    )


@pytest.mark.parametrize("buffer", [NumpyBuffer])
def test_resize(buffer):
    arr = np.array([1, 2, 3], dtype=np.float64)
    buf = buffer(arr)
    safe_buf = make_safe(buf)

    buf_stack = Stack(buf, NumpyBuffer)
    ctx = TestContext()

    # Resize to a larger size
    safe_buf.resize(5)
    assert safe_buf.length() == 5
    buf.resize(5)
    assert isinstance(safe_buf.underlying_buffer, NumpyBuffer)
    assert np.array_equal(safe_buf.underlying_buffer.arr, buf.arr)

    assert safe_buf.ftype.c_resize(ctx, buf_stack, 81) == buf.ftype.c_resize(
        ctx, buf_stack, 81
    )

    assert safe_buf.ftype.numba_resize(ctx, buf_stack, 81) == buf.ftype.numba_resize(
        ctx, buf_stack, 81
    )

    with pytest.raises(ValueError):
        safe_buf.resize(-1)
    with pytest.raises(ValueError):
        safe_buf.ftype.c_resize(ctx, buf_stack, -1)
    with pytest.raises(ValueError):
        safe_buf.ftype.numba_resize(ctx, buf_stack, -1)
