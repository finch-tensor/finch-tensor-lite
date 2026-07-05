from abc import ABC, abstractmethod
from typing import Any, TypeAlias

from finchlite.algebra.ftypes import FType

JuliaObj: TypeAlias = Any
DType = FType
number = int | float | bool | complex


def is_julia_obj(obj: Any) -> bool:
    from .julia import get_jc

    return isinstance(obj, get_jc().AnyValue)


class JLFType(FType, ABC):
    @property
    @abstractmethod
    def jl_type(self):
        pass
