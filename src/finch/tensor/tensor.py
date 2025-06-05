from abc import ABC, abstractmethod

from ..algebra import StableNumber


class AbstractTensor(ABC):
    @abstractmethod
    def shape(self): ...

    @abstractmethod
    def dtype(self): ...

    @abstractmethod
    def to_numpy(self): ...

    @abstractmethod
    def __add__(self, other): ...

    @abstractmethod
    def __mul__(self, other): ...


def fill_value(arg):
    if isinstance(arg, AbstractTensor):
        return arg.fill_value
    if isinstance(arg, StableNumber):
        return arg
    raise ValueError("Unsupported type for fill_value")
