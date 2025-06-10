from typing import Any

import numpy as np

from ...codegen import NumpyBufferFormat
from ..tensor import Level, LevelFormat

from dataclasses import dataclass, field
from ...symbolic import Format, get_format
from typing import Optional
from ... import algebra


@dataclass
class ElementLevelFormat(LevelFormat):
    fill_value: Any
    element_type: Optional[type | Format] = None
    position_type: Optional[type | Format] = None
    buffer_factory: Any = NumpyBufferFormat
    val_format: Any = None

    def __post_init__(self):
        if self.element_type is None:
            self.element_type = get_format(self.fill_value)
        if self.val_format is None:
            self.val_format = self.buffer_factory(self.element_type)
        if self.position_type is None:
            self.position_type = np.intp
        self.element_type = self.val_format.element_type
        self.fill_value = self.element_type(self.fill_value)

    def __call__(self, shape):
        """
        Creates an instance of ElementLevel with the given format.
        Args:
            fmt: The format to be used for the level.
        Returns:
            An instance of ElementLevel.
        """
        return ElementLevel(self)

    @property
    def ndims(self):
        return 0

    @property
    def shape_type(self):
        return ()

class ElementLevel(Level):
    """
    A class representing the leaf level of Finch tensors.
    """

    def __init__(self, fmt, val=None):
        """
        Initializes the ElementLevel with a format.
        Args:
            fmt: The format to be used for the level.
        """
        self.fmt = fmt
        if val is None:
            val = fmt.val_format(len=0, dtype=fmt.element_type())
        self.val = val

    @property
    def shape(self):
        return ()

    def get_format(self):
        return self.fmt
