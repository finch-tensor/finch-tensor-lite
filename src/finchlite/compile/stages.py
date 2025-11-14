from abc import abstractmethod
from .. import finch_assembly as asm
from ..symbolic import Stage
from .. import finch_notation as ntn

class NotationLowerer(Stage):
    @abstractmethod
    def __call__(self, term: ntn.Module) -> asm.Module:
        """
        Compile the given notation term into an assembly term.
        """