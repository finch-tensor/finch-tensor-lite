from dataclasses import dataclass
from typing import FrozenSet, Iterable, Any
import numpy as np

from finch.finch_notation.nodes import (
    Variable, Literal, Call, Access, Read, Update, Declare, Thaw, Freeze,
    Loop, If, Increment, Unwrap, Block, Slot, Unpack, Repack
)

from operator import add, ne

from .tensor_def import TensorDef
from .tensor_stats import TensorStats


@dataclass(frozen=True)
class DC:
    from_indices: FrozenSet[str]
    to_indices: FrozenSet[str]
    value: float


class DCStats(TensorStats):
    def __init__(self, tensor: Any, fields: Iterable[str]):
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        self.fields = list(fields)
        self.tensor = np.asarray(tensor)
        self.dcs = self._calc_dc_from_structure()

    @classmethod
    def from_tensor(cls, tensor: Any, fields: Iterable[str]) -> None:
        return None

    def _calc_dc_from_structure(self) -> set[DC]:
        ndim = self.tensor.ndim
        if ndim == 1:
            return self._vector_structure_to_dcs()
        elif ndim == 2:
            return None
        else:
            raise NotImplementedError(f"DC analysis not implemented for {ndim}D tensors")

    def _vector_structure_to_dcs(self) -> set[DC]:

        A_slot = Slot("A", type=None)
        A_unpack = Unpack(A_slot, Literal(self.tensor) )

        B_slot = Slot("B", type=None)
        B_declare = Declare(B_slot, Literal(0.0), Literal(add), ())
        B_thaw = Thaw(B_slot, Literal(add))

        body = If(
            Call(Literal(ne), Access(A_slot, Read(), Variable("i")), Literal(0.0)),
            Increment(Access(B_slot, Update(Literal(add)), ()), Literal(1.0))
        )
        loop = Loop(Variable("i"), Literal(self.tensor.shape[0]), body)

        B_freeze = Freeze(B_slot, Literal(add))
        result = Unwrap(B_slot)

        A_repack = Repack(A_slot, Literal(self.tensor))

        prog = Block.from_children(
            A_unpack,
            B_declare,
            B_thaw,
            loop,
            B_freeze,
            result,
            A_repack,
        )
        return prog


    @staticmethod
    def mapjoin(op, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def aggregate(op, dims, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def issimilar(*args, **kwargs):
        raise NotImplementedError()

    def estimate_non_fill_values(self):
        raise NotImplementedError()