from dataclasses import dataclass
from typing import Any

import numpy as np

from ...codegen import NumpyBufferFormat
from ...symbolic import Format, format
from ..fiber_tensor import Level, LevelFormat


@dataclass(unsafe_hash=True)
class ElementLevelFormat(LevelFormat):
    _fill_value: Any
    _element_type: type | Format | None = None
    _position_type: type | Format | None = None
    _buffer_factory: Any = NumpyBufferFormat
    val_format: Any = None

    def __post_init__(self):
        if self._element_type is None:
            self._element_type = format(self._fill_value)
        if self.val_format is None:
            self.val_format = self._buffer_factory(self._element_type)
        if self._position_type is None:
            self._position_type = np.intp
        self._element_type = self.val_format.element_type
        self._fill_value = self._element_type(self._fill_value)

    def __call__(self, shape=()):
        """
        Creates an instance of ElementLevel with the given format.
        Args:
            fmt: The format to be used for the level.
        Returns:
            An instance of ElementLevel.
        """
        if len(shape) != 0:
            raise ValueError("ElementLevelFormat must be called with an empty shape.")
        return ElementLevel(self)

    @property
    def ndim(self):
        return 0

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def element_type(self):
        return self._element_type

    @property
    def position_type(self):
        return self._position_type

    @property
    def shape_type(self):
        return ()

    @property
    def buffer_factory(self):
        return self._buffer_factory


def element(
    fill_value=None,
    element_type=None,
    position_type=None,
    buffer_factory=None,
    val_format=None,
):
    """
    Creates an ElementLevelFormat with the given parameters.

    Args:
        fill_value: The value to be used as the fill value for the level.
        element_type: The type of elements stored in the level.
        position_type: The type of positions within the level.
        buffer_factory: The factory used to create buffers for the level.

    Returns:
        An instance of ElementLevelFormat.
    """
    return ElementLevelFormat(
        _fill_value=fill_value,
        _element_type=element_type,
        _position_type=position_type,
        _buffer_factory=buffer_factory,
        val_format=val_format,
    )


@dataclass
class ElementLevel(Level):
    """
    A class representing the leaf level of Finch tensors.
    """

    _format: ElementLevelFormat
    val: Any | None = None

    def __post_init__(self):
        if self.val is None:
            self.val = self._format.val_format(len=0, dtype=self._format.element_type())

    @property
    def shape(self):
        return ()

    @property
    def format(self):
        return self._format
