import operator
from abc import ABC, abstractmethod

import numpy as np

ufunc_map = {
    np.add: operator.add,
    np.subtract: operator.sub,
    np.multiply: operator.mul,
    np.negative: operator.neg,
    np.positive: operator.pos,
    np.absolute: operator.abs,
    np.abs: operator.abs,
    # Add more ufuncs as needed
}


class AbstractOverrideTensor(ABC):
    @abstractmethod
    def override_module(self):
        """Return the module that implements the override logic."""

    def __array_function__(self, func, types, args, kwargs):
        """Override NumPy functions using the __array_function__ protocol."""
        # https://numpy.org/neps/nep-0018-array-function-protocol.html
        func = getattr(self.override_module(), func.__name__)
        if func is None:
            return NotImplemented
        return func(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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
        if ufunc in ufunc_map:
            if method == "__call__":
                return self.override_module().elementwise(
                    ufunc_map[ufunc], *inputs, **kwargs
                )
            if method == "reduce":
                return self.override_module().reduce(ufunc, *inputs, **kwargs)
        return NotImplemented

    def __array_namespace__(self, *, api_version=None):
        # https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html#array_api.array.__array_namespace__
        if api_version is None:
            api_version = "2024.12"

        if api_version not in {"2024.12"}:
            raise ValueError(f'"{api_version}" Array API version not supported.')
        import finch

        return finch
