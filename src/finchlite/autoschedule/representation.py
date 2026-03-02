from dataclasses import dataclass
from typing import Any

from .rw_traits import (
    Capabilities,
    RandomRead,
    RandomWrite,
    SequentialRead,
    SequentialWrite,
)


@dataclass
class ElementData:
    """Scalar element or 0-dim tensor"""

    fill_value: Any
    element_type: type

    def ndims(self) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(RandomRead, RandomWrite)


@dataclass
class SparseData:
    """Tensor where slices can be entirely fill_value"""

    lvl: Any

    def ndims(self) -> int:
        return 1 + self.lvl.ndims()

    def capabilities(self) -> Capabilities:
        return Capabilities(SequentialRead, SequentialWrite)


@dataclass
class RepeatData:
    """
    RepeatData(lvl)

    Represents a tensor `A` where `A[:, ..., :, i]` is sometimes
    entirely fill_value(lvl) and sometimes represented by repeated
    runs of `lvl`.
    """

    lvl: Any

    def ndims(self) -> int:
        return 1 + self.lvl.ndims()

    def capabilities(self) -> Capabilities:
        return Capabilities(SequentialRead, SequentialWrite)


@dataclass
class SparseRepeatData:
    """
    SparseRepeatData

    Equivalent to SparseData(RepeatData(lvl))
    """

    lvl: Any

    def ndims(self) -> int:
        return 1 + self.lvl.ndims()

    def capabilities(self) -> Capabilities:
        return Capabilities(SequentialRead, SequentialWrite)


@dataclass
class DenseData:
    """
    DenseData(lvl)

    Represents a tensor `A` where each `A[:, ..., :, i]` is represented by `lvl`.
    """

    lvl: Any

    def ndims(self) -> int:
        return 1 + self.lvl.ndims()

    def capabilities(self) -> Capabilities:
        return Capabilities(RandomRead, RandomWrite)


@dataclass
class ExtrudeData:
    """
    ExtrudeData(lvl)

    Represents a tensor `A` where `A[:, ..., :, 1]` is the only slice,
    and is represented by `lvl`.
    """

    lvl: Any

    def ndims(self) -> int:
        return 1 + self.lvl.ndims()

    def capabilities(self) -> Capabilities:
        return Capabilities(RandomRead, None)


@dataclass
class HollowData:
    """
    HollowData(lvl)

    Represents a tensor which is represented by `lvl` but is sometimes
    entirely `fill_value(lvl)`.
    """

    lvl: Any

    def ndims(self) -> int:
        return self.lvl.ndims()

    def capabilities(self) -> Capabilities:
        return self.lvl.capabilities()
