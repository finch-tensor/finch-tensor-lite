from abc import abstractmethod

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..symbolic import Stage


class NotationLowerer(Stage):
    @abstractmethod
    def transform(self, term: ntn.Module) -> tuple[asm.Module]:
        """
        Compile the given notation term into an assembly term.
        """
