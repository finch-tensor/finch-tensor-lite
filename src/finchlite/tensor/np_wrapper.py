from typing import Any

import numpy as np

from finchlite.algebra import FType, ftype, intp, normalize_device
from finchlite.algebra.tensor import Tensor, TensorFType


class NumPyFType(TensorFType):
    def __init__(self, dtype: np.dtype, ndim: int, device=None):
        self._dtype = np.dtype(dtype)
        self._ndim = ndim
        self._device = normalize_device(device)

    @property
    def fill_value(self) -> Any:
        return self._dtype.type(0)

    @property
    def element_type(self) -> FType:
        return ftype(self._dtype.type)

    @property
    def shape_type(self) -> tuple[FType, ...]:
        return (intp,) * self._ndim

    @property
    def device(self):
        return self._device

    def construct(self, shape: tuple) -> "NumPyWrapper":
        # creates a zero-filled tensor
        return NumPyWrapper(np.zeros(shape, dtype=self._dtype), device=self.device)

    def __call__(self, val: "NumPyWrapper") -> "NumPyWrapper":
        """
        Convert a tensor to this numpy tensor type.

        Args:
            val: A tensor to convert to this type.
        Returns:
            A NumPyWrapper instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )

    def __eq__(self, other):
        if not isinstance(other, NumPyFType):
            return False
        return (
            self._dtype == other._dtype
            and self._ndim == other._ndim
            and self.device == other.device
        )

    def __hash__(self):
        return hash((self._dtype, self._ndim, self.device))

    def from_numpy(self, arr: np.ndarray) -> "NumPyWrapper":
        return NumPyWrapper(arr, device=self.device)


class NumPyWrapper(Tensor):
    def __init__(self, data: np.ndarray, device=None):
        self._data = np.asarray(data)
        self._device = normalize_device(device)

    @property
    def ftype(self) -> NumPyFType:
        return NumPyFType(self._data.dtype, self._data.ndim, self._device)

    @property
    def shape(self) -> tuple:
        return self._data.shape

    @property
    def fill_value(self) -> Any:
        """Default fill value."""
        return self.ftype.fill_value

    @property
    def element_type(self) -> FType:
        """Data type of the tensor's elements."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple[FType, ...]:
        """Shape type of the tensor."""
        return self.ftype.shape_type

    @property
    def device(self):
        return self._device

    def to_device(self, device, /, *, stream=None):
        if stream is not None:
            raise ValueError(f"stream argument is not supported; got {stream!r}")
        device = normalize_device(device)
        if device == self.device:
            return self
        return NumPyWrapper(self._data, device=device)

    def item(self):
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to Python scalar.")
        return self._data.item()

    def to_numpy(self):
        return self._data

    def to_scipy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_scipy.")
