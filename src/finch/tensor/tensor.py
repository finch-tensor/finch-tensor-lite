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
    if isinstance(arg, LazyTensor):
        return arg.fill_value
    elif isinstance(arg, (int, float)):
        return arg
    else:
        raise ValueError("Unsupported type for fill_value")