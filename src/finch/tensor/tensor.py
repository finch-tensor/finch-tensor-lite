from abc import ABC, abstractmethod


class AbstractTensor(ABC):
    @abstractmethod
    def shape(self):
        pass

    @abstractmethod
    def dtype(self):
        pass

    @abstractmethod
    def to_numpy(self):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass


def fill_value(arg):
    from ..interface.lazy import LazyTensor
    from ..finch_logic import Immediate

    if isinstance(arg, LazyTensor | Immediate):
        return arg.fill_value
    if isinstance(arg, (int, float, bool, complex)):
        return arg
    raise TypeError("Unsupported type for fill_value")
