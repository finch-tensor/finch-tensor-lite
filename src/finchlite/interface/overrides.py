from abc import ABC, abstractmethod
from typing import Any

import numpy as np

import finchlite

from ..algebra import ffuncs
from ..algebra.tensor import Tensor

element_wise_ufunc_map = {
    np.add: ffuncs.add,
    np.subtract: ffuncs.sub,
    np.multiply: ffuncs.mul,
    np.negative: ffuncs.neg,
    np.positive: ffuncs.pos,
    np.absolute: ffuncs.abs,
    np.abs: ffuncs.abs,
    np.bitwise_invert: ffuncs.invert,
    np.bitwise_and: ffuncs.and_,
    np.bitwise_or: ffuncs.or_,
    np.bitwise_xor: ffuncs.xor,
    np.bitwise_left_shift: ffuncs.lshift,
    np.bitwise_right_shift: ffuncs.rshift,
    np.true_divide: ffuncs.truediv,
    np.floor_divide: ffuncs.floordiv,
    np.mod: ffuncs.mod,
    np.pow: ffuncs.pow,
    np.reciprocal: ffuncs.reciprocal,
    np.sin: ffuncs.sin,
    np.sinh: ffuncs.sinh,
    np.cos: ffuncs.cos,
    np.cosh: ffuncs.cosh,
    np.tan: ffuncs.tan,
    np.tanh: ffuncs.tanh,
    np.asin: ffuncs.asin,
    np.asinh: ffuncs.asinh,
    np.acos: ffuncs.acos,
    np.acosh: ffuncs.acosh,
    np.atan: ffuncs.atan,
    np.atanh: ffuncs.atanh,
    np.hypot: ffuncs.hypot,
    np.atan2: ffuncs.atan2,
    np.floor: ffuncs.floor,
    np.ceil: ffuncs.ceil,
    np.trunc: ffuncs.trunc,
    np.exp: ffuncs.exp,
    np.expm1: ffuncs.expm1,
    np.log: ffuncs.log,
    np.log1p: ffuncs.log1p,
    np.log2: ffuncs.log2,
    np.log10: ffuncs.log10,
    np.logaddexp: ffuncs.logaddexp,
    np.signbit: ffuncs.signbit,
    np.copysign: ffuncs.copysign,
    np.nextafter: ffuncs.nextafter,
    np.sqrt: ffuncs.sqrt,
    np.square: ffuncs.square,
    np.sign: ffuncs.sign,
    np.isfinite: ffuncs.isfinite,
    np.isinf: ffuncs.isinf,
    np.isnan: ffuncs.isnan,
    np.logical_and: ffuncs.logical_and,
    np.logical_or: ffuncs.logical_or,
    np.logical_xor: ffuncs.logical_xor,
    np.logical_not: ffuncs.logical_not,
    np.equal: ffuncs.equal,
    np.not_equal: ffuncs.not_equal,
    np.less: ffuncs.less,
    np.less_equal: ffuncs.less_equal,
    np.greater: ffuncs.greater,
    np.greater_equal: ffuncs.greater_equal,
}

ufunc_map: dict[Any, Any] = {
    np.matmul: "matmul",
}


class OverrideTensor(Tensor, ABC):
    @abstractmethod
    def override_module(self):
        """Return the module that implements the override logic."""
        raise NotImplementedError("No module override defined.")

    def __array_function__(self, func, types, args, kwargs):
        """Override NumPy functions using the __array_function__ protocol."""
        # https://numpy.org/neps/nep-0018-array-function-protocol.html
        func = getattr(self.override_module(), func.__name__)
        if func is None:
            return NotImplemented
        return func(*args, **kwargs)

    def __array_ufunc__(self, ufunc: np.ufunc, method, *inputs, **kwargs):
        """Override NumPy ufuncs using the __array_ufunc__ protocol."""
        # https://numpy.org/devdocs/user/basics.ufuncs.html#ufuncs-basics
        # https://numpy.org/devdocs/reference/ufuncs.html#ufuncs-methods
        if kwargs.get("out") is not None:
            raise NotImplementedError("out parameter is not supported")
        if kwargs.get("where") is not None:
            raise NotImplementedError("where parameter is not supported")
        if kwargs.get("casting") is not None:
            raise NotImplementedError("casting parameter is not supported")
        if kwargs.get("order") is not None:
            raise NotImplementedError("order parameter is not supported")
        if kwargs.get("axes") is not None:
            kwargs["axis"] = kwargs.pop("axes")
        if ufunc in element_wise_ufunc_map:
            if method == "__call__":
                return self.override_module().elementwise(
                    element_wise_ufunc_map[ufunc], *inputs, **kwargs
                )
            if method == "reduce":
                return self.override_module().reduce(ufunc, *inputs, **kwargs)
        if ufunc in ufunc_map:
            func_name = ufunc_map[ufunc]
            if method == "__call__":
                return getattr(self.override_module(), func_name)(*inputs, **kwargs)
        return NotImplemented

    def __array_namespace__(self, *, api_version=None):
        # https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html#array_api.array.__array_namespace__
        if api_version is None:
            api_version = "2024.12"

        if api_version not in {"2024.12"}:
            raise ValueError(f'"{api_version}" Array API version not supported.')

        return finchlite
