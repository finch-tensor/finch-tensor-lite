from __future__ import annotations

from abc import ABC, abstractmethod

from finchlite.algebra import Tensor


class AccessCapability:
    pass


class Sequential(AccessCapability):
    pass


class Random(AccessCapability):
    pass


class FormatProperty:
    pass


class Dense(FormatProperty):
    pass


class Sparse(FormatProperty):
    pass


class Blocked(FormatProperty):
    pass


class Repeated(FormatProperty):
    pass


class Extruded(FormatProperty):
    pass


class Hollow(FormatProperty):
    pass


class AxiomaticTensor(Tensor, ABC):
    @property
    @abstractmethod
    def format_properties(self) -> list[tuple[type[FormatProperty], ...]]:
        ...
