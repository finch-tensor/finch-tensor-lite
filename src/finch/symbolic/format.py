from abc import ABC, abstractmethod

from ..algebra import query_property


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


def get_format(x):
    """
    Get the format of `x`.
    """
    if hasattr(x, "get_format"):
        return x.get_format()
    try:
        return query_property(
            x,
            "get_format",
            "__attr__",
        )
    except AttributeError:
        return type(x)
