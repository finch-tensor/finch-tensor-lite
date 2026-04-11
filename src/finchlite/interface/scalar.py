from __future__ import annotations

from typing import Any

from ..algebra import FType, TensorFType, ftype
from .eager import EagerTensor


class ScalarFType(TensorFType):
    def __init__(self, _element_type: FType, _fill_value: Any):
        self._element_type = _element_type
        self._fill_value = _fill_value

    def __eq__(self, other):
        if isinstance(other, ScalarFType):
            return (
                self._element_type == other._element_type
                and self._fill_value == other._fill_value
            )
        return False

    def __hash__(self):
        return hash((self._element_type, self._fill_value))

    def __call__(self, shape: tuple) -> Scalar:
        if shape != ():
            raise ValueError("ScalarFType can only be called with empty shape ()")
        return self._element_type(self._fill_value)

    def from_numpy(self, arr):
        return self(arr)

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def element_type(self) -> FType:
        return self._element_type

    @property
    def shape_type(self):
        return ()

    def lower_unwrap(self, ctx, obj):
        return obj.obj


class Scalar(EagerTensor):
    def __init__(self, val: Any, fill_value: Any = None):
        if fill_value is None:
            fill_value = val
        self.val = val
        self._fill_value = fill_value

    @property
    def ftype(self):
        return ScalarFType(type(self.val), self._fill_value)

    @property
    def shape(self):
        return ()

    @property
    def fill_value(self) -> Any:
        """Default value to fill the scalar."""
        return self.ftype.fill_value

    @property
    def element_type(self) -> FType:
        """Data type of the scalar."""
        return self.ftype.element_type

    @property
    def shape_type(self) -> tuple:
        """Shape type of the scalar."""
        return self.ftype.shape_type

    def __getitem__(self, idx):
        return self.val

    def __str__(self):
        return str(self.val)

    def to_numpy(self):
        return self.val
