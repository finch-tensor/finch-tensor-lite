from abc import ABC, abstractmethod

from ..symbolic import FType, FTyped


class Dict(FTyped, ABC):
    """
    Abstract base class for a map data structure.
    Hash tables should be such that their bucket size can be resized, with Tree
    maps turning that into a no-op.
    """

    @abstractmethod
    def __init__(
        self, key_len: int, value_len: int, map: "dict[tuple,tuple] | None"
    ): ...

    @property
    @abstractmethod
    def ftype(self) -> "DictFType": ...

    @property
    def value_type(self):
        """
        Return type of values stored in the hash table
        (probably some TupleFType)
        """
        return self.ftype.value_type

    @property
    def key_type(self):
        """
        Return type of keys stored in the hash table
        (probably some TupleFType)
        """
        return self.ftype.key_type

    @abstractmethod
    def load(self, idx: tuple):
        """
        Method to access some element in the map. Will panic if the key doesn't exist.
        """
        ...

    @abstractmethod
    def exists(self, idx: tuple) -> bool:
        """
        Method to check if the element exists in the map.
        """
        ...

    @abstractmethod
    def store(self, idx: tuple, val):
        """
        Method to store elements in the map. Ideally it should just create new
        elements.
        """
        ...


class DictFType(FType):
    """
    Abstract base class for an ftype corresponding to a map.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Create an instance of an object in this ftype with the given arguments.
        """
        ...

    @property
    @abstractmethod
    def value_type(self):
        """
        Return the type of elements stored in the map.
        This is typically the same as the dtype used to create the map.
        """
        ...

    @property
    @abstractmethod
    def key_type(self):
        """
        Returns the type used for the length of the map.
        """
        ...
