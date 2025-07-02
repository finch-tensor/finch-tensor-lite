from collections import OrderedDict
from typing import Any, Iterable, Mapping

from finch.algebra import fill_value

class TensorDef:
    def __init__(
        self,
        index_set: Iterable[str],
        dim_sizes: Mapping[str, float],
        fill_val: Any,
    ):
        self.index_set = set(index_set)
        self.dim_sizes = OrderedDict(dim_sizes)
        self.fill_val = fill_val

    def copy(self) -> "TensorDef":
        """
        Deep Copy of TensorDef fields
        """
        return TensorDef(
            index_set = self.index_set.copy(),
            dim_sizes      = self.dim_sizes.copy(),
            fill_val       = self.fill_val,
        )