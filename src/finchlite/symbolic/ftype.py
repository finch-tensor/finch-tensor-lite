from abc import ABC, abstractmethod

from ..algebra import query_property


class FType(ABC):
    @abstractmethod
    def __eq__(self, other): ...

    @abstractmethod
    def __hash__(self): ...

    def fisinstance(self, other):
        """
        Check if `other` is an instance of this ftype.
        """
        return ftype(other) == self


class FTyped:
    """
    Abstract base class for objects that can be formatted.
    """

    @property
    @abstractmethod
    def ftype(self):
        """
        The ftype of the object.
        """
        ...


def fisinstance(x, f):
    """
    Check if `x` is an instance of `f`.
    """
    if isinstance(f, type):
        return isinstance(x, f)
    return f.fisinstance(x)


def ftype(x):
    """
    Get the ftype of `x`.
    """
    if hasattr(x, "ftype"):
        return x.ftype
    try:
        return query_property(
            x,
            "ftype",
            "__attr__",
        )
    except AttributeError:
        return type(x)
