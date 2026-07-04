from . import dtypes
from .typing import DType


class __array_namespace_info__:
    def capabilities(self) -> dict[str, bool]:
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
        }

    def default_device(self) -> str:
        return "cpu"

    def default_dtypes(self, *, device: str | None = None) -> dict[str, DType]:
        if device not in ["cpu", None]:
            raise ValueError(
                f'Device not understood. Only "cpu" is allowed, but received: {device}'
            )
        return {
            "real floating": dtypes.float64,
            "complex floating": dtypes.complex128,
            "integral": dtypes.int_,
            "indexing": dtypes.int_,
        }

    _bool_dtypes = {"bool": dtypes.bool}
    _signed_integer_dtypes = {
        "int8": dtypes.int8,
        "int16": dtypes.int16,
        "int32": dtypes.int32,
        "int64": dtypes.int64,
    }
    _unsigned_integer_dtypes = {
        "uint8": dtypes.uint8,
        "uint16": dtypes.uint16,
        "uint32": dtypes.uint32,
        "uint64": dtypes.uint64,
    }
    _real_floating_dtypes = {
        "float32": dtypes.float32,
        "float64": dtypes.float64,
    }
    _complex_floating_dtypes = {
        "complex64": dtypes.complex64,
        "complex128": dtypes.complex128,
    }

    def dtypes(
        self,
        *,
        device: str | None = None,
        kind: str | tuple[str, ...] | None = None,
    ) -> dict[str, DType]:
        if device not in ["cpu", None]:
            raise ValueError(
                f'Device not understood. Only "cpu" is allowed, but received: {device}'
            )
        if kind is None:
            return (
                self._bool_dtypes
                | self._signed_integer_dtypes
                | self._unsigned_integer_dtypes
                | self._real_floating_dtypes
                | self._complex_floating_dtypes
            )
        if kind == "bool":
            return self._bool_dtypes
        if kind == "signed integer":
            return self._signed_integer_dtypes
        if kind == "unsigned integer":
            return self._unsigned_integer_dtypes
        if kind == "integral":
            return self._signed_integer_dtypes | self._unsigned_integer_dtypes
        if kind == "real floating":
            return self._real_floating_dtypes
        if kind == "complex floating":
            return self._complex_floating_dtypes
        if kind == "numeric":
            return (
                self._signed_integer_dtypes
                | self._unsigned_integer_dtypes
                | self._real_floating_dtypes
                | self._complex_floating_dtypes
            )
        if isinstance(kind, tuple):
            res = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    def devices(self) -> list[str]:
        return ["cpu"]
