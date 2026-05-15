from abc import abstractmethod

from finchlite import finch_assembly as asm
from finchlite import finch_notation as ntn
from finchlite.symbolic import Stage


class NotationLowerer(Stage):
    @abstractmethod
    def __call__(self, term: ntn.Module) -> asm.Module:
        """
        Compile the given notation term into an assembly term.
        """
