import operator
import sys
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


class OverrideTensor(Tensor):
    def mod(self):
        return sys.modules["finchlite"]

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

    def __add__(self, other):
        return self.mod.add(self, other)

    def __radd__(self, other):
        return self.mod.add(other, self)

    def __sub__(self, other):
        return self.mod.subtract(self, other)

    def __rsub__(self, other):
        return self.mod.subtract(other, self)

    def __mul__(self, other):
        return self.mod.multiply(self, other)

    def __rmul__(self, other):
        return self.mod.multiply(other, self)

    def __abs__(self):
        return self.mod.abs(self)

    def __pos__(self):
        return self.mod.positive(self)

    def __neg__(self):
        return self.mod.negative(self)

    def __invert__(self):
        return self.mod.bitwise_inverse(self)

    def __and__(self, other):
        return self.mod.bitwise_and(self, other)

    def __rand__(self, other):
        return self.mod.bitwise_and(other, self)

    def __lshift__(self, other):
        return self.mod.bitwise_left_shift(self, other)

    def __rlshift__(self, other):
        return self.mod.bitwise_left_shift(other, self)

    def __or__(self, other):
        return self.mod.bitwise_or(self, other)

    def __ror__(self, other):
        return self.mod.bitwise_or(other, self)

    def __rshift__(self, other):
        return self.mod.bitwise_right_shift(self, other)

    def __rrshift__(self, other):
        return self.mod.bitwise_right_shift(other, self)

    def __xor__(self, other):
        return self.mod.bitwise_xor(self, other)

    def __rxor__(self, other):
        return self.mod.bitwise_xor(other, self)

    def __truediv__(self, other):
        return self.mod.truediv(self, other)

    def __rtruediv__(self, other):
        return self.mod.truediv(other, self)

    def __floordiv__(self, other):
        return self.mod.floordiv(self, other)

    def __rfloordiv__(self, other):
        return self.mod.floordiv(other, self)

    def __mod__(self, other):
        return self.mod.mod(self, other)

    def __rmod__(self, other):
        return self.mod.mod(other, self)

    def __pow__(self, other):
        return self.mod.power(self, other)

    def __rpow__(self, other):
        return self.mod.power(other, self)

    def __matmul__(self, other):
        return self.mod.matmul(self, other)

    def __rmatmul__(self, other):
        return self.mod.matmul(other, self)

    def __sin__(self):
        return self.mod.sin(self)

    def __sinh__(self):
        return self.mod.sinh(self)

    def __cos__(self):
        return self.mod.cos(self)

    def __cosh__(self):
        return self.mod.cosh(self)

    def __tan__(self):
        return self.mod.tan(self)

    def __tanh__(self):
        return self.mod.tanh(self)

    def __asin__(self):
        return self.mod.asin(self)

    def __asinh__(self):
        return self.mod.asinh(self)

    def __acos__(self):
        return self.mod.acos(self)

    def __acosh__(self):
        return self.mod.acosh(self)

    def __atan__(self):
        return self.mod.atan(self)

    def __atanh__(self):
        return self.mod.atanh(self)

    def __atan2__(self, other):
        return self.mod.atan2(self, other)

    def __complex__(self):
        """
        Converts a zero-dimensional array to a Python `complex` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to complex.")
        # dispatch to the scalar value's `__complex__` method
        return complex(self[()])

    def __float__(self):
        """
        Converts a zero-dimensional array to a Python `float` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to float.")
        # dispatch to the scalar value's `__float__` method
        return float(self[()])

    def __int__(self):
        """
        Converts a zero-dimensional array to a Python `int` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to int.")
        # dispatch to the scalar value's `__int__` method
        return int(self[()])

    def __bool__(self):
        """
        Converts a zero-dimensional array to a Python `bool` object.
        """
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to bool.")
        # dispatch to the scalar value's `__bool__` method
        return bool(self[()])

    def __index__(self) -> int:
        if self.ndim != 0:
            raise ValueError("Cannot convert non-scalar tensor to index.")
        return operator.index(self.__int__())

    def __log__(self):
        return self.mod.log(self)

    def __log1p__(self):
        return self.mod.log1p(self)

    def __log2__(self):
        return self.mod.log2(self)

    def __log10__(self):
        return self.mod.log10(self)

    def __logaddexp__(self, other):
        return self.mod.logaddexp(self, other)

    def __logical_and__(self, other):
        return self.mod.logical_and(self, other)

    def __logical_or__(self, other):
        return self.mod.logical_or(self, other)

    def __logical_xor__(self, other):
        return self.mod.logical_xor(self, other)

    def __logical_not__(self):
        return self.mod.logical_not(self)

    def __lt__(self, other):
        return self.mod.less(self, other)

    def __le__(self, other):
        return self.mod.less_equal(self, other)

    def __gt__(self, other):
        return self.mod.greater(self, other)

    def __ge__(self, other):
        return self.mod.greater_equal(self, other)

    def __eq__(self, other):
        return self.mod.equal(self, other)

    def __ne__(self, other):
        return self.mod.not_equal(self, other)
