from abc import ABC, abstractmethod


class Format(ABC):
    @abstractmethod
    def __eq__(self, other): ...

    @abstractmethod
    def __hash__(self): ...

    def has_format(self, other):
        """
        Check if `other` is an instance of this format.
        """
        return other.get_format() == self


class Formattable(ABC):
    @abstractmethod
    def get_format(self):
        """
        Get the format of the object.
        """
        ...


def has_format(x, f):
    """
    Check if `x` is an instance of `f`.
    """
    if isinstance(f, type):
        return isinstance(x, f)
    return f.has_format(x)
