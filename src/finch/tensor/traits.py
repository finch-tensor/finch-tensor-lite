from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from finch.algebra import Tensor


class AccessCapability:
    """Marker for a way a level can be read or written."""


class Sequential(AccessCapability):
    """The level supports ordered traversal."""


class Random(AccessCapability):
    """The level supports direct access by index."""


class FormatProperty:
    """Marker for a structural property exposed by a tensor format."""


@dataclass(frozen=True)
class Dense(FormatProperty):
    """
    Every slice identified by ``dims`` contains a non-fill value.

    Equivalently, the tensor's non-fill support has a complete projection onto
    these dimensions.
    """

    dims: tuple[int, ...]


@dataclass(frozen=True)
class Blocked(FormatProperty):
    """
    Each non-fill slice at an odd position along the conclusion dimensions
    occurs together with a subsequent non-fill slice.
    """

    hypothesis_dims: tuple[int, ...]
    conclusion_dims: tuple[int, ...]


@dataclass(frozen=True)
class Repeated(FormatProperty):
    """
    Each slice at an odd position along the conclusion dimension occurs together
    with an subsequent identical slice.
    """

    hypothesis_dims: tuple[int, ...]
    conclusion_dims: tuple[int, ...]


class AxiomaticTensor(Tensor, ABC):
    @property
    @abstractmethod
    def level_format_properties(self) -> list[FormatProperty]: ...
