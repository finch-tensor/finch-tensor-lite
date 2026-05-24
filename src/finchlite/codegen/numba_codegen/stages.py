from abc import abstractmethod

from finchlite import finch_assembly as asm
from finchlite.symbolic import Stage


class NumbaCode:
    def __init__(self, code: str):
        self.code = code

    def __str__(self) -> str:
        return self.code


class NumbaLowerer(Stage):
    @abstractmethod
    def __call__(self, prgm: asm.Module) -> NumbaCode:
        """
        Lower the given assembly program to Numba code.
        """


__all__ = ["NumbaCode", "NumbaLowerer"]
