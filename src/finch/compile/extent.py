from ..finch_notation.interpreter import HaltState
from .lower import lower_looplets

from dataclasses import dataclass
from typing import Any



@dataclass(eq=True, frozen=True)
class ExtentFormat:
    start: Any
    end: Any

    def lower_loop(self, ctx, idx, body):
        # Lower the loop using the extent format
        return Extent(self.start_type(0), self.end_type(0)).loop(ctx, idx, body)


def extent(start, end):
    """
    Create an extent value for a loop.
    """
    return Extent(start, end)

def dimension(tns, mode):
    end = tns.shape[mode]
    return extent(type(end)(0), end)

