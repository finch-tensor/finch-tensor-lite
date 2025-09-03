from dataclasses import dataclass
from typing import Any

import numpy as np

from ...codegen import NumpyBufferFType
from ...symbolic import FType, ftype
from ..fiber_tensor import Level, LevelFType


@dataclass(unsafe_hash=True)
class ElementLevelFType(LevelFType):
    _fill_value: Any
    _element_type: type | FType | None = None
    _position_type: type | FType | None = None
    _buffer_factory: Any = NumpyBufferFType
    val_format: Any = None

    def __post_init__(self):
        if self._element_type is None:
            self._element_type = ftype(self._fill_value)
        if self.val_format is None:
            self.val_format = self._buffer_factory(self._element_type)
        if self._position_type is None:
            self._position_type = np.intp
        self._element_type = self.val_format.element_type
        self._fill_value = self._element_type(self._fill_value)

    def __call__(self, shape=()):
        """
        Creates an instance of ElementLevel with the given ftype.
        Args:
            fmt: The ftype to be used for the level.
        Returns:
            An instance of ElementLevel.
        """
        if len(shape) != 0:
            raise ValueError("ElementLevelFType must be called with an empty shape.")
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
    Creates an ElementLevelFType with the given parameters.

    Args:
        fill_value: The value to be used as the fill value for the level.
        element_type: The type of elements stored in the level.
        position_type: The type of positions within the level.
        buffer_factory: The factory used to create buffers for the level.

    Returns:
        An instance of ElementLevelFType.
    """
    return ElementLevelFType(
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

    _format: ElementLevelFType
    val: Any | None = None

    def __post_init__(self):
        if self.val is None:
            self.val = self._format.val_format(len=0, dtype=self._format.element_type())

    @property
    def shape(self):
        return ()

    @property
    def ftype(self):
        return self._format
