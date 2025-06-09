from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from finch.symbolic import Format, Formattable


class LevelFormat(Format, ABC):
    """
    An abstract base class representing the format of levels.
    """

    @abstractmethod
    def ndims(self):
        """
        Returns the number of dimensions of the fibers in the structure.
        """
        ...

    @abstractmethod
    def fill_value(self):
        """
        Returns the fill value of the fibers, or `None` if the fill_value is dynamic.
        """
        ...

    @abstractmethod
    def element_type(self):
        """
        Returns the type of elements stored in the fibers.
        """
        ...

    @abstractmethod
    def shape_type(self):
        """
        Returns the type of the shape of the fibers.
        """
        ...

    @abstractmethod
    def position_type(self):
        """
        Returns the type of positions within the levels.
        """
        ...

    @abstractmethod
    def buffer_format(self):
        """
        Returns the format of the buffer used for the fibers.
        This is typically a NumpyBufferFormat or similar.
        """
        ...


class Level(Formattable, ABC):
    """
    An abstract base class representing a fiber allocator that manages fibers in
    a tensor.
    """

    @abstractmethod
    def shape(self):
        """
        Returns the shape of the fibers in the structure.
        """
        ...

    def ndims(self):
        return self.get_format().ndims()

    def fill_value(self):
        return self.get_format().fill_value()

    def element_type(self):
        return self.get_format().element_type()

    def shape_type(self):
        return self.get_format().shape_type()

    def position_type(self):
        return self.get_format().position_type()

    def buffer_format(self):
        return self.get_format().buffer_format()


Tp = TypeVar("Tp")


@dataclass
class FiberTensor[Tp](Formattable):
    """
    A class representing a tensor with fiber structure.

    Attributes:
        lvl: a fiber allocator that manages the fibers in the tensor.
    """

    lvl: Level
    pos: Tp | None = None

    def __init__(self, lvl, pos=None):
        """
        Initializes the FiberTensor with a level object `lvl`.
        Args:
            lvl: a fiber allocator that manages the fibers in the tensor.
        """
        self.lvl = lvl
        self.pos = pos

    def __repr__(self):
        res = f"FiberTensor(lvl={self.lvl}"
        if self.pos is not None:
            res += f", pos={self.pos}"
        res += ")"
        return res


class FiberTensorFormat(Format, ABC):
    """
    An abstract base class representing the format of a fiber tensor.

    Attributes:
        lvl: a fiber allocator that manages the fibers in the tensor.
    """

    lvl: LevelFormat

    def __init__(self, lvl):
        """
        Initializes the FiberTensorFormat with a level object `lvl`.
        Args:
            lvl: a fiber allocator that manages the fibers in the tensor.
        """
        self.lvl = lvl

    def __call__(self, shape):
        """
        Creates an instance of a FiberTensor with the given arguments.
        """
        return FiberTensor(self.lvl, shape)
