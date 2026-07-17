from __future__ import annotations

from abc import ABC, abstractmethod

from finchlite.algebra import Tensor


class AccessCapability:
    pass


class Sequential(AccessCapability):
    pass


class Random(AccessCapability):
    pass


@dataclass
class FormatProperty:
    hypothesis_dims:list[int]
    conclusion_dim:list[int]

"""
The dense(x -> y) format property says that if there exist indices z such that
T[x, z] != 0, then T[x, y, z] != 0 for all y.
"""
class Dense(FormatProperty):
    pass


class Blocked(FormatProperty):
    pass


class Repeated(FormatProperty):
    pass


class Extruded(FormatProperty):
    pass


class AxiomaticTensor(Tensor, ABC):
    @property
    @abstractmethod
    def format_properties(self) -> list[FormatProperty]:
        ...