from .buffer import Buffer, BufferFormat


class SafeBuffer(Buffer):
    """
    A wrapper buffer that adds bounds checking to any underlying buffer. Ensures
    safe access to buffer elements by checking indices before load/store operations.
    """

    def __init__(self, underlying_buffer: Buffer):
        """
        Initialize SafeBuffer with an underlying buffer to wrap.

        Args:
            underlying_buffer: The buffer to add safety checks to
        """
        if not isinstance(underlying_buffer, Buffer):
            raise TypeError(f"Expected Buffer instance, got {type(underlying_buffer)}")
        self._underlying_buffer = underlying_buffer

    @property
    def format(self):
        """
        Returns the format of the safe buffer.
        """
        return SafeBufferFormat(self._underlying_buffer.format)

    def length(self):
        """
        Return the length of the underlying buffer.
        """
        return self._underlying_buffer.length()

    def _check_bounds(self, idx: int):
        """
        Check if the given index is within bounds.

        Args:
            idx: The index to check

        Raises:
            IndexError: If index is out of bounds
            TypeError: If index is not an integer
        """
        if not isinstance(idx, int):
            raise TypeError(f"Buffer indices must be integers, got {type(idx)}")

        buffer_length = self.length()
        if idx < 0:
            # Handle negative indexing
            idx = buffer_length + idx

        if idx < 0 or idx >= buffer_length:
            raise IndexError(
                f"index {idx} is out of bounds for buffer of length {buffer_length}"
            )

        return idx

    def load(self, idx: int):
        """
        Load an element from the buffer with bounds checking.

        Args:
            idx: The index to load from

        Returns:
            The element at the given index

        Raises:
            IndexError: If index is out of bounds
        """
        checked_idx = self._check_bounds(idx)
        return self._underlying_buffer.load(checked_idx)

    def store(self, idx: int, val):
        """
        Store an element in the buffer with bounds checking.

        Args:
            idx: The index to store at
            val: The value to store

        Raises:
            IndexError: If index is out of bounds
        """
        checked_idx = self._check_bounds(idx)
        self._underlying_buffer.store(checked_idx, val)

    def resize(self, new_length: int):
        """
        Resize the underlying buffer.

        Args:
            new_length: The new length for the buffer

        Raises:
            ValueError: If new_length is negative
            TypeError: If new_length is not an integer
        """
        if not isinstance(new_length, int):
            raise TypeError(f"Buffer length must be an integer, got {type(new_length)}")

        if new_length < 0:
            raise ValueError(f"Buffer length must be non-negative, got {new_length}")

        self._underlying_buffer.resize(new_length)

    @property
    def underlying_buffer(self):
        """
        Provide access to the underlying buffer for advanced use cases.
        """
        return self._underlying_buffer


class SafeBufferFormat(BufferFormat):
    """
    A format for SafeBuffer that wraps another buffer format.
    """

    def __init__(self, underlying_format: BufferFormat):
        """
        Initialize with the underlying buffer format to wrap.

        Args:
            underlying_format: The buffer format to add safety to
        """
        if not isinstance(underlying_format, BufferFormat):
            raise TypeError(
                f"Expected BufferFormat instance, got {type(underlying_format)}"
            )
        self._underlying_format = underlying_format

    def __eq__(self, other):
        if not isinstance(other, SafeBufferFormat):
            return False
        return self._underlying_format == other._underlying_format

    def __hash__(self):
        return hash(("SafeBufferFormat", self._underlying_format))

    @property
    def element_type(self):
        """
        Return the element type of the underlying format.
        """
        return self._underlying_format.element_type

    @property
    def length_type(self):
        """
        Return the length type of the underlying format.
        """
        return self._underlying_format.length_type

    def __call__(self, *args, **kwargs):
        """
        Create a SafeBuffer wrapping a buffer created by the underlying format.

        Args:
            *args: Arguments to pass to the underlying format
            **kwargs: Keyword arguments to pass to the underlying format

        Returns:
            A SafeBuffer instance wrapping the created buffer
        """
        underlying_buffer = self._underlying_format(*args, **kwargs)
        return SafeBuffer(underlying_buffer)

    @property
    def underlying_format(self):
        """
        Provide access to the underlying format.
        """
        return self._underlying_format


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
        >>> buf = NumpyBuffer(arr)
        >>> # Make it safe
        >>> safe_buf = make_safe(buf)
        >>> # Now bounds checking is automatic
        >>> safe_buf.load(0)  # OK
        >>> safe_buf.load(10)  # Raises IndexError
    """
    return SafeBuffer(buffer)
