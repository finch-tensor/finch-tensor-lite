from abc import ABC, abstractmethod

from .. import finch_logic as lgc
from .. import finch_notation as ntn


class LogicNotationLowerer(ABC):
    @abstractmethod
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, lgc.TableValueFType]
    ) -> ntn.Module:
        """
        Generate Finch Notation from the given logic and input types.  Also
        return a dictionary including additional tables needed to run the kernel.
        """
