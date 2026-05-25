from abc import abstractmethod

from finchlite import finch_assembly as asm
from finchlite.symbolic import Stage


class CCode:
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


__all__ = ["CCode", "CLowerer"]
