from dataclasses import dataclass
from typing import FrozenSet, Iterable, Any
import numpy as np

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
            return self._matrix_structure_to_dcs()
        else:
            raise NotImplementedError(f"DC analysis not implemented for {ndim}D tensors")

    def _vector_structure_to_dcs(self) -> set[DC]:
        i = self.fields[0]
        nnz = float(np.count_nonzero(self.tensor))

        return {
            DC(frozenset(), frozenset([i]), nnz)
        }

    def _matrix_structure_to_dcs(self) -> set[DC]:
        i, j = self.fields
        A = self.tensor

        row_nnz = np.count_nonzero(A, axis=1)
        col_nnz = np.count_nonzero(A, axis=0)

        return {
            DC(frozenset(), frozenset([i, j]), float(np.count_nonzero(A))),
            DC(frozenset(), frozenset([i]), float(np.count_nonzero(row_nnz > 0))),
            DC(frozenset(), frozenset([j]), float(np.count_nonzero(col_nnz > 0))),
            DC(frozenset([i]), frozenset([i, j]), float(np.max(row_nnz))),
            DC(frozenset([j]), frozenset([i, j]), float(np.max(col_nnz))),
        }

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