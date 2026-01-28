from typing import Any

import numpy as np

from ..algebra.tensor import Tensor, TensorFType


class NumPyFType(TensorFType):
    def __init__(self, dtype: np.dtype, ndim: int):
        self._dtype = np.dtype(dtype)
        self._ndim = ndim

    @property
    def fill_value(self) -> Any:
        return self._dtype.type(0)

    @property
    def element_type(self) -> Any:
        return self._dtype

    @property
    def shape_type(self) -> tuple[type, ...]:
        return (int,) * self._ndim

    def __call__(self, shape: tuple) -> "NumPyWrapper":
        # creates a zero-filled tensor
        return NumPyWrapper(np.zeros(shape, dtype=self._dtype))

    def __eq__(self, other):
        if not isinstance(other, NumPyFType):
            return False
        return self._dtype == other._dtype and self._ndim == other._ndim

    def __hash__(self):
        return hash((self._dtype, self._ndim))


class NumPyWrapper(Tensor):
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data)
        self._ftype = NumPyFType(self._data.dtype, self._data.ndim)

    @property
    def ftype(self) -> NumPyFType:
        return self._ftype

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def fill_value(self) -> Any:
        """Default fill value."""
        return self.ftype.fill_value

    @property
    def element_type(self) -> Any:
        """Data type of the tensor's elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple[type, ...]:
        """Shape type of the tensor."""
        return self.ftype.shape_type
