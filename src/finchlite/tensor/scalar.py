from __future__ import annotations

from typing import Any

import numpy as np

from finchlite.algebra import FType, TensorFType, ffuncs, ftype, normalize_device
from finchlite.algebra.ftypes import FDType

from .override_tensor import OverrideTensor


class ScalarFType(TensorFType):
    def __init__(self, _element_type: FType, _fill_value: Any, _device=None):
        elt = _element_type
        if not isinstance(elt, FDType):
            raise TypeError(f"Scalar element type must be FDType, got {elt}")
        self._element_type = elt
        self._fill_value = _fill_value
        self._device = normalize_device(_device)

    def __eq__(self, other):
        if isinstance(other, ScalarFType):
            return (
                self._element_type == other._element_type
                and ffuncs.same(self._fill_value, other._fill_value)
                and self.device == other.device
            )
        return False

    def __hash__(self):
        return hash(
            (self._element_type, ffuncs.samehash(self._fill_value), self.device)
        )

    def construct(self, shape: tuple) -> Scalar:
        if shape != ():
            raise ValueError("ScalarFType can only be called with empty shape ()")
        return self._element_type(self._fill_value)

    def __call__(self, val: Any) -> Scalar:
        """
        Convert a tensor to this scalar tensor type.

        Args:
            val: A value to convert to this type.
        Returns:
            A Scalar instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )

    def from_numpy(self, arr):
        return self(arr)

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def device(self):
        return self._device

    @property
    def element_type(self) -> FType:
        return self._element_type

    @property
    def shape_type(self):
        return ()

    def lower_unwrap(self, ctx, obj):
        return obj.obj


class Scalar(OverrideTensor):
    def __init__(self, val: Any, fill_value: Any = None, device=None):
        if fill_value is None:
            fill_value = val
        self.val = val
        self._fill_value = fill_value
        self._device = normalize_device(device)

    @property
    def ftype(self):
        return ScalarFType(ftype(self.val), self._fill_value, self._device)

    @property
    def shape(self):
        return ()

    @property
    def fill_value(self) -> Any:
        """Default value to fill the scalar."""
        return self.ftype.fill_value

    @property
    def device(self):
        return self._device

    def to_device(self, device, /, *, stream=None):
        if stream is not None:
            raise ValueError(f"stream argument is not supported; got {stream!r}")
        device = normalize_device(device)
        if device == self.device:
            return self
        return Scalar(self.val, fill_value=self._fill_value, device=device)

    @property
    def element_type(self) -> FType:
        """Data type of the scalar."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple:
        """Shape type of the scalar."""
        return self.ftype.shape_type

    def item(self):
        return self.val.item() if hasattr(self.val, "item") else self.val

    def __array__(self, dtype=None, copy=None):
        if copy is None:
            return np.asarray(self.val, dtype=dtype)
        return np.array(self.val, dtype=dtype, copy=copy)

    def __getitem__(self, idx):
        if idx == () or idx is Ellipsis or idx == (...,):
            return self
        raise IndexError("Too many indices for scalar tensor.")

    def __str__(self):
        return str(self.val)

    def to_numpy(self):
        return self.val
