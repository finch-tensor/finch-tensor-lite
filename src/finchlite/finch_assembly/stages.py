from abc import ABC, abstractmethod
from typing import Any
from ..symbolic import Stage
from . import nodes as asm

class AssemblyKernel(ABC):
    """
    Represents a callable assembly kernel.
    """
    @abstractmethod
    def __call__(self, *args) -> Any:
        ...

class AssemblyLibrary(ABC):
    """
    Represents a module containing assembly kernels.
    """
    @abstractmethod
    def __getattr__(self, name: str) -> AssemblyKernel:
        """
        Get the assembly Kernel corresponding to the given name.
        """
        ...

class AssemblyLoader(Stage):
    @abstractmethod
    def __call__(self, term: asm.Module) -> AssemblyLibrary:
        """
        Load the given assembly program into a runnable module.
        """
    
class AssemblyTransform(Stage):
    @abstractmethod
    def __call__(self, term: asm.AssemblyNode) -> asm.AssemblyNode:
        """
        Transform the given assembly term into another assembly term.
        """