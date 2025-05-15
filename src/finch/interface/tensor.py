from abc import ABC, abstractmethod
from . import lazy
from .fuse import *

class EagerTensor(ABC):
    @abstractmethod
    def shape(self):
        """Return the shape of the tensor."""
        pass

    @abstractmethod
    def dtype(self):
        """Return the data type of the tensor."""
        pass

    @abstractmethod
    def to_numpy(self):
        """Convert the tensor to a NumPy array."""
        pass

    @abstractmethod
    def __add__(self, other):
        compute(lazy.lazy(self).__add__(other))

    @abstractmethod
    def __mul__(self, other):
        """Define multiplication for tensors."""
        pass

def prod(arr, /, axis=None):
    if arr.is_lazy():
        return lazy.prod(arr, axis=axis)
    else:
        return compute(lazy.prod(lazy.lazy(arr), axis=axis))
