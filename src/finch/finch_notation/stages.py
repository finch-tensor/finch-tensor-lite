from abc import abstractmethod

from finch import finch_assembly as asm
from finch.symbolic import Stage

from . import nodes as ntn


class NotationLoader(Stage):
    @abstractmethod
    def lower(self, term: ntn.Module) -> asm.AssemblyLibrary:
        """
        Load the given notation program into a runnable module.
        """


class NotationTransform(Stage):
    @abstractmethod
    def lower(self, term: ntn.Module) -> ntn.Module:
        """
        Transform the given assembly term into another assembly term.
        """
