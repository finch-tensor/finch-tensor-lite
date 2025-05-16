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

# def sum(arr, /, axis=None):
#     if arr.is_lazy():
#         return lazy.sum(arr, axis=axis)
#     else:
#         return compute(lazy.sum(lazy.lazy(arr), axis=axis))

# def any(arr, /, axis=None):
#     if arr.is_lazy():
#         return lazy.any(arr, axis=axis)
#     else:
#         return compute(lazy.any(lazy.lazy(arr), axis=axis))

# def all(arr, /, axis=None):
#     if arr.is_lazy():
#         return lazy.all(arr, axis=axis)
#     else:
#         return compute(lazy.all(lazy.lazy(arr), axis=axis))

# def min(arr, /, axis=None):
#     if arr.is_lazy():
#         return lazy.min(arr, axis=axis)
#     else:
#         return compute(lazy.min(lazy.lazy(arr), axis=axis))

# def max(arr, /, axis=None):
#     if arr.is_lazy():
#         return lazy.max(arr, axis=axis)
#     else:
#         return compute(lazy.max(lazy.lazy(arr), axis=axis))

# def mean(arr, /, axis=None):
#     if arr.is_lazy():
#         return lazy.mean(arr, axis=axis)
#     else:
#         return compute(lazy.mean(lazy.lazy(arr), axis=axis))

# def std(arr, /, axis=None):
#     if arr.is_lazy():
#         return lazy.std(arr, axis=axis)
#     else:
#         return compute(lazy.std(lazy.lazy(arr), axis=axis))

# def var(arr, /, axis=None):
#     if arr.is_lazy():
#         return lazy.std(arr, axis=axis)
#     else:
#         return compute(lazy.std(lazy.lazy(arr), axis=axis))

# def argmin(arr, /, axis=None):
#     if arr.is_lazy():
#         return lazy.argmin(arr, axis=axis)
#     else:
#         return compute(lazy.argmin(lazy.lazy(arr), axis=axis))

# def argmax(arr, /, axis=None):
#     if arr.is_lazy():
#         return lazy.argmin(arr, axis=axis)
#     else:
#         return compute(lazy.argmin(lazy.lazy(arr), axis=axis))
