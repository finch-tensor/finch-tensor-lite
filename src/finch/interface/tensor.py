from abc import ABC, abstractmethod

from . import lazy
from .fuse import compute


class EagerTensor(ABC):
    @abstractmethod
    def shape(self):
        """Return the shape of the tensor."""

    @abstractmethod
    def dtype(self):
        """Return the data type of the tensor."""

    @abstractmethod
    def to_numpy(self):
        """Convert the tensor to a NumPy array."""

    @abstractmethod
    def __add__(self, other):
        compute(lazy.lazy(self).__add__(other))

    @abstractmethod
    def __mul__(self, other):
        """Define multiplication for tensors."""


def prod(arr, /, axis=None):
    if arr.is_lazy():
        return lazy.prod(arr, axis=axis)
    return compute(lazy.prod(lazy.lazy(arr), axis=axis))
