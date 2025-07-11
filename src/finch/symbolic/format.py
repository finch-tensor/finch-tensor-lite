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
        return format(other) == self


class Formattable:
    """
    Abstract base class for objects that can be formatted.
    """

    @property
    @abstractmethod
    def format(self):
        """
        The format of the object.
        """
        ...


def has_format(x, f):
    """
    Check if `x` is an instance of `f`.
    """
    if isinstance(f, type):
        return isinstance(x, f)
    return f.has_format(x)


def format(x):
    """
    Get the format of `x`.
    """
    if hasattr(x, "format"):
        return x.format
    try:
        return query_property(
            x,
            "format",
            "__attr__",
        )
    except AttributeError:
        return type(x)
