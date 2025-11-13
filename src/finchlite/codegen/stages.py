from abc import abstractmethod

from finchlite.symbolic import Stage

from .. import finch_assembly as asm


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
    def __call__(self, prgm: asm.Module) -> CCode:
        """
        Lower the given assembly program to C code.
        """


class NumbaLowerer(Stage):
    @abstractmethod
    def __call__(self, prgm: asm.Module) -> NumbaCode:
        """
        Lower the given assembly program to Numba code.
        """
