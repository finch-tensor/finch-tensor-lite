from ..finch_assembly.buffer import Buffer
from .c import CBufferFType, CStackFType
from .numba_backend import NumbaBufferFType, NumbaStackFType


def _check_bounds(buf_length, idx):
    """
    Check if the given idx is within bounds of the buffer

    Args:
        buf_length: The buffer length
        idx: The idx to check

    Raises:
        idxError: If idx is out of bounds
    """
    idx = int(idx)

    if idx < 0 or idx >= buf_length:
        raise IndexError(
            f"idx {idx} is out of bounds for buffer of length {buf_length}"
        )


def _check_resize(new_length):
    if not isinstance(new_length, int):
        raise TypeError(f"Buffer length must be an integer, got {type(new_length)}")
    if new_length < 0:
        raise ValueError(f"Buffer length must be non-negative, got {new_length}")


class SafeBuffer(Buffer):
    """
    A wrapper buffer that adds bounds checking to any underlying buffer. Ensures
    safe access to buffer elements by checking indices before load/store operations.
    """

    def __init__(self, buf: Buffer):
        """
        Initialize SafeBuffer with an underlying buffer to wrap.

        Args:
            underlying_buffer: The buffer to add safety checks to
        """
        if not isinstance(buf, Buffer):
            raise TypeError(f"Expected Buffer instance, got {type(buf)}")
        self._buf = buf

    @property
    def ftype(self):
        """
        Returns the format of the safe buffer.
        """
        return SafeBufferFType(self._buf.ftype)

    def length(self):
        """
        Return the length of the underlying buffer.
        """
        return self._buf.length()

    def load(self, idx: int):
        """
        Load an element from the buffer with bounds checking.

        Args:
            idx: The idx to load from

        Returns:
            The element at the given idx

        Raises:
            idxError: If idx is out of bounds
        """
        _check_bounds(self.length(), idx)
        return self._buf.load(idx)

    def store(self, idx: int, val):
        """
        Store an element in the buffer with bounds checking.

        Args:
            idx: The idx to store at
            val: The value to store

        Raises:
            idxError: If idx is out of bounds
        """
        _check_bounds(self.length(), idx)
        self._buf.store(idx, val)

    def resize(self, new_length: int):
        """
        Resize the underlying buffer.

        Args:
            new_length: The new length for the buffer

        Raises:
            ValueError: If new_length is negative
            TypeError: If new_length is not an integer
        """
        _check_resize(new_length)
        self._buf.resize(new_length)

    @property
    def underlying_buffer(self):
        """
        Provide access to the underlying buffer for advanced use cases.
        """
        return self._buf


class SafeBufferFType(CBufferFType, NumbaBufferFType, CStackFType, NumbaStackFType):
    """
    A buffer format that adds safety checks to an underlying buffer format.
    """

    def __init__(self, underlying_format):
        """
        Initialize with the underlying buffer format to wrap.

        Args:
            underlying_format: The buffer format to add safety to
        """
        # TODO: Uncomment and enforce these requirements
        # required_types = (
        #     # BufferFType,
        #     CBufferFType,
        #     NumbaBufferFType,
        #     CStackFType,
        #     NumbaStackFType,
        # )
        # missing_types = [
        #     req_type.__name__
        #     for req_type in required_types
        #     if not isinstance(underlying_format, req_type)
        # ]

        # if missing_types:
        #     raise TypeError(
        #         f"SafeBufferFType requires underlying buffer type to "
        #         f"implement all of {[t.__name__ for t in required_types]}"
        #         f", but {type(underlying_format).__name__} "
        #         f"is missing: {missing_types}"
        #     )

        self._ftype = underlying_format

    def __eq__(self, other):
        if not isinstance(other, SafeBufferFType):
            return False
        return self.underlying_ftype == other.underlying_ftype

    def __hash__(self):
        return hash(("SafeBufferFType", self.underlying_ftype))

    @property
    def element_type(self):
        """
        Return the element type of the underlying format.
        """
        return self.underlying_ftype.element_type

    @property
    def length_type(self):
        """
        Return the length type of the underlying format.
        """
        return self.underlying_ftype.length_type

    def __call__(self, *args, **kwargs):
        """
        Create a SafeBuffer wrapping a buffer created by the underlying format.

        Args:
            *args: Arguments to pass to the underlying format
            **kwargs: Keyword arguments to pass to the underlying format

        Returns:
            A SafeBuffer instance wrapping the created buffer
        """
        underlying_buffer = self.underlying_ftype(*args, **kwargs)
        return SafeBuffer(underlying_buffer)

    @property
    def underlying_ftype(self):
        """
        Provide access to the underlying format.
        """
        return self._ftype

    # =====================================================
    # =========== C Buffer Type Implementation ============
    # =====================================================

    def c_length(self, ctx, buffer):
        return self.underlying_ftype.c_length(ctx, buffer)

    def c_type(self):
        return self.underlying_ftype.c_type()

    def c_load(self, ctx, buffer, idx):
        # An expression in the target language is expected to be returned
        # ! BAD: This assumes the underlying type is an integer type
        # ! BAD: Magically returns -1 on out-of-bounds, which may be valid data
        return (
            f"({self.c_length(ctx, buffer)} > {ctx(idx)} && {ctx(idx)} >= 0) ?"
            "{self.underlying_ftype.c_load(ctx, buffer, idx)} : -1"
        )

    def c_store(self, ctx, buffer, idx, value):
        return (
            f"if({self.c_length(ctx, buffer)} > {ctx(idx)} && {ctx(idx)} >= 0)"
            "{" + self.underlying_ftype.c_store(ctx, buffer, idx, value) + "}"
        )

    def c_resize(self, ctx, buffer, new_len):
        return (
            f"if({new_len} >= 0) {{"
            + self.underlying_ftype.c_resize(ctx, buffer, new_len)
            + "}"
        )

    def serialize_to_c(self, obj):
        return self.underlying_ftype.serialize_to_c(obj)

    def deserialize_from_c(self, obj, c_buffer):
        return self.underlying_ftype.deserialize_from_c(obj, c_buffer)

    def construct_from_c(self, res):
        return self.underlying_ftype.construct_from_c(res)

    # =====================================================
    # =========== C Stack Type Implementation =============
    # =====================================================

    def c_unpack(self, ctx, lhs, rhs):
        return self._ftype.c_unpack(ctx, lhs, rhs)

    def c_repack(self, ctx, lhs, rhs):
        return self.underlying_ftype.c_repack(ctx, lhs, rhs)

    # =====================================================
    # ========= Numba Buffer Type Implementation ==========
    # =====================================================

    def numba_type(self):
        return self.underlying_ftype.numba_type()

    def numba_length(self, ctx, buffer):
        return self.underlying_ftype.numba_length(ctx, buffer)

    def numba_load(self, ctx, buffer, idx):
        ctx_2 = ctx.subblock()
        self.underlying_ftype.numba_load(ctx_2, buffer, idx)
        body_code = ctx_2.emit()
        ctx.exec(
            f"{ctx.feed}if {self.numba_length(ctx, buffer)} > {ctx(idx)}"
            f"and {ctx(idx)} >= 0:"
            f"\n{body_code}"
        )

    def numba_store(self, ctx, buffer, idx, value=None):
        ctx_2 = ctx.subblock()
        self.underlying_ftype.numba_store(ctx_2, buffer, idx, value)
        body_code = ctx_2.emit()
        ctx.exec(
            f"{ctx.feed}if {self.numba_length(ctx, buffer)} > {ctx(idx)}"
            f"and {ctx(idx)} >= 0:"
            f"\n{body_code}"
        )

    def numba_resize(self, ctx, buffer, new_len):
        ctx_2 = ctx.subblock()
        self.underlying_ftype.numba_resize(ctx_2, buffer, new_len)
        body_code = ctx_2.emit()
        ctx.exec(f"{ctx.feed}if {new_len} >= 0:\n{body_code}")

    def numba_unpack(self, ctx, var_n, val):
        return self.underlying_ftype.numba_unpack(ctx, var_n, val)

    def numba_repack(self, ctx, lhs, obj):
        return self.underlying_ftype.numba_repack(ctx, lhs, obj)

    def serialize_to_numba(self, obj):
        return self.underlying_ftype.serialize_to_numba(obj)

    def deserialize_from_numba(self, obj, numba_buffer):
        return self.underlying_ftype.deserialize_from_numba(obj, numba_buffer)

    def construct_from_numba(self, numba_buffer):
        return self.underlying_ftype.construct_from_numba(numba_buffer)


def make_safe(buffer: Buffer) -> SafeBuffer:
    """
    Convenience function to wrap any buffer in a SafeBuffer.

    Args:
        buffer: The buffer to make safe

    Returns:
        A SafeBuffer wrapping the input buffer

    Example:
        >>> import numpy as np
        >>> from finch.codegen import NumpyBuffer
        >>> from finch.finch_assembly.safe_buffer import make_safe
        >>> # Create a regular buffer
        >>> arr = np.array([1, 2, 3, 4])
        >>> buffer = NumpyBuffer(arr)
        >>> # Make it safe
        >>> safe_buffer = make_safe(buffer)
        >>> # Now bounds checking is automatic
        >>> safe_buf.load(0)  # OK
        >>> safe_buf.load(10)  # Raises idxError
    """
    return SafeBuffer(buffer)


def safe_wrapper(buffer_type):
    """
    Returns a version of buffer constructor that returns
    the buffer wrapped in a SafeBuffer
    """

    def _make_safe(*args):
        return make_safe(buffer_type(*args))

    return _make_safe
