from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

from finch.symbolic import Format, Formattable


class LevelFormat(Format, ABC):
    """
    An abstract base class representing the format of levels.

    Subclasses must define the following properties:
    - `ndims`: Number of dimensions of the fibers in the structure.
    - `fill_value`: Fill value of the fibers, or `None` if dynamic.
    - `element_type`: Type of elements stored in the fibers.
    - `shape_type`: Type of the shape of the fibers.
    - `position_type`: Type of positions within the levels.
    - `buffer_factory`: Function to create default buffers for the fibers.
    """


class Level(Formattable, ABC):
    """
    An abstract base class representing a fiber allocator that manages fibers in
    a tensor.

    Subclasses must define the following properties:
    - `shape`: The shape of the fibers in the structure.
    """

    @property
    def ndims(self):
        return self.format.ndims

    @property
    def fill_value(self):
        return self.format.fill_value

    @property
    def element_type(self):
        return self.format.element_type

    @property
    def shape_type(self):
        return self.format.shape_type

    @property
    def position_type(self):
        return self.format.position_type

    @property
    def buffer_factory(self):
        return self.format.buffer_factory


Tp = TypeVar("Tp")


@dataclass
class FiberTensor(Generic[Tp], Formattable):
    """
    A class representing a tensor with fiber structure.

    Attributes:
        lvl: a fiber allocator that manages the fibers in the tensor.
    """

    lvl: Level
    pos: Tp

    def __repr__(self):
        res = f"FiberTensor(lvl={self.lvl}"
        if self.pos is not None:
            res += f", pos={self.pos}"
        res += ")"
        return res

    @property
    def format(self):
        """
        Returns the format of the fiber tensor, which is a FiberTensorFormat.
        """
        return FiberTensorFormat(self.lvl.format, type(self.pos))

    @property
    def shape(self):
        return self.lvl.shape

    @property
    def ndims(self):
        return self.lvl.ndims

    @property
    def shape_type(self):
        return self.lvl.shape_type

    @property
    def element_type(self):
        return self.lvl.element_type

    @property
    def fill_value(self):
        return self.lvl.fill_value

    @property
    def position_type(self):
        return self.lvl.position_type

    @property
    def buffer_factory(self):
        """
        Returns the format of the buffer used for the fibers.
        This is typically a NumpyBufferFormat or similar.
        """
        return self.lvl.buffer_factory


@dataclass
class FiberTensorFormat(Format, ABC):
    """
    An abstract base class representing the format of a fiber tensor.

    Attributes:
        lvl: a fiber allocator that manages the fibers in the tensor.
    """

    lvl: LevelFormat
    pos: type | None = None

    def __post_init__(self):
        if self.pos is None:
            self.pos = self.lvl.position_type

    def __call__(self, shape):
        """
        Creates an instance of a FiberTensor with the given arguments.
        """
        return FiberTensor(self.lvl(shape), self.lvl.position_type(1))

    @property
    def shape(self):
        return self.lvl.shape

    @property
    def ndims(self):
        return self.lvl.ndims

    @property
    def shape_type(self):
        return self.lvl.shape_type

    @property
    def element_type(self):
        return self.lvl.element_type

    @property
    def fill_value(self):
        return self.lvl.fill_value

    @property
    def position_type(self):
        return self.lvl.position_type

    @property
    def buffer_factory(self):
        """
        Returns the format of the buffer used for the fibers.
        This is typically a NumpyBufferFormat or similar.
        """
        return self.lvl.buffer_factory
