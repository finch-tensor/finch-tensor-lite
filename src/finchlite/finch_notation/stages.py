from abc import abstractmethod

from .. import finch_assembly as asm
from ..symbolic import Stage
from . import nodes as ntn


class NotationLoader(Stage):
    @abstractmethod
    def __call__(self, term: ntn.Module) -> asm.AssemblyLibrary:
        """
        Load the given notation program into a runnable module.
        """


class NotationTransform(Stage):
    @abstractmethod
    def __call__(self, term: ntn.Module) -> ntn.Module:
        """
        Transform the given assembly term into another assembly term.
        """


class OptNotationLoader(NotationLoader):
    def __init__(self, *opts: NotationTransform, ctx: NotationLoader):
        self.ctx = ctx
        self.opts = opts

    def __call__(self, term: ntn.Module) -> asm.AssemblyLibrary:
        for opt in self.opts:
            term = opt(term)
        return self.ctx(term)
