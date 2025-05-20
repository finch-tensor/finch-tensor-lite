from abc import ABC, abstractmethod

from ..algebra import StableNumber


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
    if isinstance(arg, AbstractTensor):
        return arg.fill_value
    if isinstance(arg, StableNumber):
        return arg
    raise ValueError("Unsupported type for fill_value")
