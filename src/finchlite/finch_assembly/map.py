from abc import ABC, abstractmethod
from ..symbolic import FType, FTyped


class Map(FTyped, ABC):
    """
    Abstract base class for a map data structure.
    Hash tables should be such that their bucket size can be resized, with Tree
    maps turning that into a no-op.
    """

    @abstractmethod
    def __init__(self, length: int, dtype: type): ...

    @property
    def element_type(self):
        """
        Return the type of elements stored in the hash table.
        This is typically the same as the dtype used to create the map.
        """
        return self.ftype.element_type()

    @property
    def length_type(self):
        """
        Return the type of indices used to access elements in the hash table.
        This is typically an integer type.
        """
        return self.ftype.length_type()

    @abstractmethod
    def load(self, idx1: int, idx2: int):
        """
        Method to access some element in the map. Will panic if the key doesn't exist.
        """
        ...

    @abstractmethod
    def exists(self, idx1: int, idx2: int) -> bool:
        """
        Method to check if the element exists in the map.
        """
        ...

    @abstractmethod
    def store(self, idx1: int, idx2: int, val):
        """
        Method to store elements in the map. Ideally it should just create new
        elements.
        """
        ...


class MapFType(FType):
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
    def element_type(self):
        """
        Return the type of elements stored in the map.
        This is typically the same as the dtype used to create the map.
        """
        ...

    @property
    def length_type(self):
        """
        Returns the type used for the length of the map.
        """
        return int
