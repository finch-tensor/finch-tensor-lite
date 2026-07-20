from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from finchlite.algebra import Tensor


class AccessCapability:
    """Marker for a way a level can be read or written."""


class Sequential(AccessCapability):
    """The level supports ordered traversal."""


class Random(AccessCapability):
    """The level supports direct access by index."""


@dataclass(frozen=True)
class FormatProperty:
    """
    A structural rule from known dimensions to implied dimensions.

    For example, a dense property with hypothesis dims ``x`` and conclusion
    dims ``y`` says that any non-fill slice at ``x`` contains all ``y`` values.
    """

    hypothesis_dims: tuple[int, ...]
    conclusion_dims: tuple[int, ...]


class Dense(FormatProperty):
    """
    Every value along the conclusion dimensions exists whenever the hypothesis
    dimensions identify a non-fill slice.
    """


class Blocked(FormatProperty):
    """
    Adjacent values in the conclusion dimensions tend to occur together when the
    hypothesis dimensions are fixed.
    """


class Repeated(FormatProperty):
    """
    Adjacent values in the conclusion dimensions tend to share a value when the
    hypothesis dimensions are fixed.
    """


class Extruded(FormatProperty):
    """The conclusion dimensions are represented by one repeated slice."""


class AxiomaticTensor(Tensor, ABC):
    @property
    @abstractmethod
    def level_format_properties(self) -> list[FormatProperty]: ...
