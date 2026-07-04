from abc import ABC, abstractmethod

import juliacall as jc
from finchlite.algebra.ftypes import FType

JuliaObj = jc.AnyValue
DType = FType
number = int | float | bool | complex


class JLFType(FType, ABC):
    @property
    @abstractmethod
    def jl_type(self):
        pass
