from abc import abstractmethod

from finchlite import finch_assembly as asm
from finchlite.symbolic import Stage


class CCode:
    def __init__(self, code: str):
        self.code = code

    def __str__(self) -> str:
        return self.code


class NumbaCode:
    def __init__(self, code: str):
        self.code = code

    def __str__(self) -> str:
        return self.code


class CLowerer(Stage):
    @abstractmethod
    def transform(self, prgm: asm.Module) -> tuple[CCode]:
        """
        Lower the given assembly program to C code.
        """


class NumbaLowerer(Stage):
    @abstractmethod
    def transform(self, prgm: asm.Module) -> tuple[NumbaCode]:
        """
        Lower the given assembly program to Numba code.
        """
