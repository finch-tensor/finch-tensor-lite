from dataclasses import dataclass
from typing import Any

import numpy as np

from ...codegen import NumpyBufferFormat
from ...symbolic import Format, format
from ..tensor import Level, LevelFormat


@dataclass
class ElementLevelFormat(LevelFormat):
    fill_value: Any
    element_type: type | Format | None = None
    position_type: type | Format | None = None
    buffer_factory: Any = NumpyBufferFormat
    val_format: Any = None

    def __post_init__(self):
        if self.element_type is None:
            self.element_type = format(self.fill_value)
        if self.val_format is None:
            self.val_format = self.buffer_factory(self.element_type)
        if self.position_type is None:
            self.position_type = np.intp
        self.element_type = self.val_format.element_type
        self.fill_value = self.element_type(self.fill_value)

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
    def ndims(self):
        return 0

    @property
    def shape_type(self):
        return ()


@dataclass
class ElementLevel(Level):
    """
    A class representing the leaf level of Finch tensors.
    """

    format: ElementLevelFormat
    val: Any | None = None

    def __post_init__(self):
        if self.val is None:
            self.val = self.format.val_format(len=0, dtype=self.format.element_type())

    @property
    def shape(self):
        return ()
