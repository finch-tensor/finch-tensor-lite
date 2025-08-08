from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from finch.algebra import Tensor, TensorFType
from finch.symbolic import FType, FTyped


class LevelFType(FType, ABC):
    """
    An abstract base class representing the ftype of levels.
    """

    @property
    @abstractmethod
    def ndim(self):
        """
        Number of dimensions of the fibers in the structure.
        """
        ...

    @property
    @abstractmethod
    def fill_value(self):
        """
        Fill value of the fibers, or `None` if dynamic.
        """
        ...

    @property
    @abstractmethod
    def element_type(self):
        """
        Type of elements stored in the fibers.
        """
        ...

    @property
    @abstractmethod
    def shape_type(self):
        """
        Tuple of types of the dimensions in the shape
        """
        ...

    @property
    @abstractmethod
    def position_type(self):
        """
        Type of positions within the levels.
        """
        ...

    @property
    @abstractmethod
    def buffer_factory(self):
        """
        Function to create default buffers for the fibers.
        """
        ...


class Level(FTyped, ABC):
    """
    An abstract base class representing a fiber allocator that manages fibers in
    a tensor.
    """

    @property
    @abstractmethod
    def shape(self):
        """
        Shape of the fibers in the structure.
        """
        ...

    @property
    def ndim(self):
        return self.ftype.ndim

    @property
    def fill_value(self):
        return self.ftype.fill_value

    @property
    def element_type(self):
        return self.ftype.element_type

    @property
    def shape_type(self):
        return self.ftype.shape_type

    @property
    def position_type(self):
        return self.ftype.position_type

    @property
    def buffer_factory(self):
        return self.ftype.buffer_factory


Tp = TypeVar("Tp")


@dataclass
class FiberTensor(Generic[Tp], Tensor):
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
    def ftype(self):
        """
        Returns the ftype of the fiber tensor, which is a FiberTensorFType.
        """
        return FiberTensorFType(self.lvl.ftype, type(self.pos))

    @property
    def shape(self):
        return self.lvl.shape

    @property
    def ndim(self):
        return self.lvl.ndim

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
        Returns the ftype of the buffer used for the fibers.
        This is typically a NumpyBufferFType or similar.
        """
        return self.lvl.buffer_factory


@dataclass(unsafe_hash=True)
class FiberTensorFType(TensorFType):
    """
    An abstract base class representing the ftype of a fiber tensor.

    Attributes:
        lvl: a fiber allocator that manages the fibers in the tensor.
    """

    lvl: LevelFType
    _position_type: type | None = None

    def __post_init__(self):
        if self._position_type is None:
            self._position_type = self.lvl.position_type

    def __call__(self, shape):
        """
        Creates an instance of a FiberTensor with the given arguments.
        """
        return FiberTensor(self.lvl(shape), self.lvl.position_type(1))

    @property
    def shape(self):
        return self.lvl.shape

    @property
    def ndim(self):
        return self.lvl.ndim

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
        return self._position_type

    @property
    def buffer_factory(self):
        """
        Returns the ftype of the buffer used for the fibers.
        This is typically a NumpyBufferFType or similar.
        """
        return self.lvl.buffer_factory


def tensor(lvl: LevelFType, position_type: type | None = None):
    """
    Creates a FiberTensorFType with the given level ftype and position type.

    Args:
        lvl: The level ftype to be used for the tensor.
        pos_type: The type of positions within the tensor. Defaults to None.

    Returns:
        An instance of FiberTensorFType.
    """
    # mypy does not understand that dataclasses generate __hash__ and __eq__
    return FiberTensorFType(lvl, position_type)  # type: ignore[abstract]
