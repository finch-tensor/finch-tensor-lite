from abc import abstractmethod

from finchlite import finch_assembly as asm
from finchlite.symbolic import Stage

from . import nodes as ntn


class NotationLoader(Stage):
    @abstractmethod
    def transform(self, term: ntn.Module) -> tuple:
        """
        Load the given notation program into a runnable module.
        """


class NotationTransform(Stage):
    @abstractmethod
    def transform(self, term: ntn.Module) -> tuple[ntn.Module]:
        """
        Transform the given assembly term into another assembly term.
        """
