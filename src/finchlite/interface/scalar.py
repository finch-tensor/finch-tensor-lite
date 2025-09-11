from __future__ import annotations

from typing import Any

from ..algebra import TensorFType, register_property
from .eager import EagerTensor


class ScalarFType(TensorFType):
    def __init__(self, _element_type: type, _fill_value: Any):
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

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def element_type(self):
        return self._element_type

    @property
    def shape_type(self):
        return ()


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

    def __getitem__(self, idx):
        return self.val


register_property(object, "asarray", "__attr__", lambda x: Scalar(x))
register_property(Scalar, "asarray", "__attr__", lambda x: x)
