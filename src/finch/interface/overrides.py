import operator
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..algebra.tensor import Tensor

element_wise_ufunc_map = {
    np.add: operator.add,
    np.subtract: operator.sub,
    np.multiply: operator.mul,
    np.negative: operator.neg,
    np.positive: operator.pos,
    np.absolute: operator.abs,
    np.abs: operator.abs,
    np.bitwise_invert: operator.invert,
    np.bitwise_and: operator.and_,
    np.bitwise_or: operator.or_,
    np.bitwise_xor: operator.xor,
    np.bitwise_left_shift: operator.lshift,
    np.bitwise_right_shift: operator.rshift,
    np.true_divide: operator.truediv,
    np.floor_divide: operator.floordiv,
    np.mod: operator.mod,
    np.pow: operator.pow,
    np.sin: np.sin,
    np.sinh: np.sinh,
    np.cos: np.cos,
    np.cosh: np.cosh,
    np.tan: np.tan,
    np.tanh: np.tanh,
    np.asin: np.asin,
    np.asinh: np.asinh,
    np.acos: np.acos,
    np.acosh: np.acosh,
    np.atan: np.atan,
    np.atanh: np.atanh,
    np.atan2: np.atan2,
    np.log: np.log,
    np.log1p: np.log1p,
    np.log2: np.log2,
    np.log10: np.log10,
    np.logaddexp: np.logaddexp,
    np.logical_and: np.logical_and,
    np.logical_or: np.logical_or,
    np.logical_xor: np.logical_xor,
    np.logical_not: np.logical_not,
    # Add more ufuncs as needed
}

ufunc_map: dict[Any, Any] = {
    np.matmul: "matmul",
}


class OverrideTensor(Tensor, ABC):
    @abstractmethod
    def override_module(self):
        """Return the module that implements the override logic."""
        ...

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
        import finch

        return finch
