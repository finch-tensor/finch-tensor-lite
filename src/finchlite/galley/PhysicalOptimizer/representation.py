from dataclasses import dataclass
from typing import Any


@dataclass
class ElementData:
    """ Scalar element or 0-dim tensor""" 
    fill_value: Any
    element_type: type

    def ndims(self) -> int:
        return 0


@dataclass
class SparseData:
    """ Tensor where slices can be entirely fill_value """
    lvl: Any

    def ndims(self) -> int:
        return 1 + self.lvl.ndims()

@dataclass
class RepeatData:
    """
    RepeatData(lvl)

    Represents a tensor `A` where `A[:, ..., :, i]` is sometimes entirely fill_value(lvl)
    and is sometimes represented by repeated runs of `lvl`.
    """
    lvl: Any

    def ndims(self) -> int:
        return 1 + self.lvl.ndims()

@dataclass
class DenseData:
    """
    DenseData(lvl)

    Represents a tensor `A` where each `A[:, ..., :, i]` is represented by `lvl`.
    """
    lvl: Any

    def ndims(self) -> int:
        return 1 + self.lvl.ndims()


@dataclass
class ExtrudeData:
    """
    ExtrudeData(lvl)

    Represents a tensor `A` where `A[:, ..., :, 1]` is the only slice, and is represented by `lvl`.
    """
    lvl: Any

    def ndims(self) -> int:
        return 1 + self.lvl.ndims()

@dataclass
class HollowData:
    """
    HollowData(lvl)

    Represents a tensor which is represented by `lvl` but is sometimes entirely `fill_value(lvl)`.
    """
    lvl: Any

    def ndims(self) -> int:
        return 1 + self.lvl.ndims()