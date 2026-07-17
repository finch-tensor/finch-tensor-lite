from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from finchlite.algebra import Tensor


class AccessCapability:
    """How a level can be read or written."""

    pass


class Sequential(AccessCapability):
    """Supports ordered traversal only."""

    pass


class Random(AccessCapability):
    """Supports direct access by index."""

    pass


@dataclass
class FormatProperty:
    """A structural fact relating one group of level dimensions to another."""

    hypothesis_dims: list[int]
    conclusion_dim: list[int]


class Dense(FormatProperty):
    """
    If some entry exists for the hypothesis dimensions, every value along the
    conclusion dimension exists.
    """

    pass


class Blocked(FormatProperty):
    """Keeping the hypothesis dimensions fixed, adjacent values in the
    conclusion dimension are likely to occur together."""

    pass


class Repeated(FormatProperty):
    """Keeping the hypothesis dimensions fixed, adjacent values in the
    conclusion dimension are likely to have the same value."""

    pass


class Extruded(FormatProperty):
    """The conclusion dimensions are represented by a single repeated slice."""

    pass


class AxiomaticTensor(Tensor, ABC):
    @property
    @abstractmethod
    def format_properties(self) -> list[FormatProperty]:
        ...
