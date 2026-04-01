from abc import ABC, abstractmethod
from typing import Any

import numpy as np

import finchlite

from ..algebra.tensor import Tensor
from ..algebra import ffunc

element_wise_ufunc_map = {
    np.add: ffunc.add,
    np.subtract: ffunc.sub,
    np.multiply: ffunc.mul,
    np.negative: ffunc.neg,
    np.positive: ffunc.pos,
    np.absolute: ffunc.abs,
    np.abs: ffunc.abs,
    np.bitwise_invert: ffunc.invert,
    np.bitwise_and: ffunc.and_,
    np.bitwise_or: ffunc.or_,
    np.bitwise_xor: ffunc.xor,
    np.bitwise_left_shift: ffunc.lshift,
    np.bitwise_right_shift: ffunc.rshift,
    np.true_divide: ffunc.truediv,
    np.floor_divide: ffunc.floordiv,
    np.mod: ffunc.mod,
    np.pow: ffunc.pow,
    np.reciprocal: ffunc.reciprocal,
    np.sin: ffunc.sin,
    np.sinh: ffunc.sinh,
    np.cos: ffunc.cos,
    np.cosh: ffunc.cosh,
    np.tan: ffunc.tan,
    np.tanh: ffunc.tanh,
    np.asin: ffunc.asin,
    np.asinh: ffunc.asinh,
    np.acos: ffunc.acos,
    np.acosh: ffunc.acosh,
    np.atan: ffunc.atan,
    np.atanh: ffunc.atanh,
    np.hypot: ffunc.hypot,
    np.atan2: ffunc.atan2,
    np.floor: ffunc.floor,
    np.ceil: ffunc.ceil,
    np.trunc: ffunc.trunc,
    np.exp: ffunc.exp,
    np.expm1: ffunc.expm1,
    np.log: ffunc.log,
    np.log1p: ffunc.log1p,
    np.log2: ffunc.log2,
    np.log10: ffunc.log10,
    np.logaddexp: ffunc.logaddexp,
    np.signbit: ffunc.signbit,
    np.copysign: ffunc.copysign,
    np.nextafter: ffunc.nextafter,
    np.sqrt: ffunc.sqrt,
    np.square: ffunc.square,
    np.sign: ffunc.sign,
    np.isfinite: ffunc.isfinite,
    np.isinf: ffunc.isinf,
    np.isnan: ffunc.isnan,
    np.logical_and: ffunc.logical_and,
    np.logical_or: ffunc.logical_or,
    np.logical_xor: ffunc.logical_xor,
    np.logical_not: ffunc.logical_not,
    np.equal: ffunc.equal,
    np.not_equal: ffunc.not_equal,
    np.less: ffunc.less,
    np.less_equal: ffunc.less_equal,
    np.greater: ffunc.greater,
    np.greater_equal: ffunc.greater_equal,
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
