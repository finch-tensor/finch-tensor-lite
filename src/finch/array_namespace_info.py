from __future__ import annotations

import builtins
from typing import Any

from finch.algebra.devices import AbstractDevice, cpu, normalize_device, serial
from finch.algebra.ftypes import (
    bool,
    complex64,
    complex128,
    float,
    float16,
    float32,
    float64,
    int,
    int8,
    int16,
    int32,
    int64,
    intp,
    isdtype,
    uint8,
    uint16,
    uint32,
    uint64,
)

_DTYPES = {
    "bool": bool,
    "int8": int8,
    "int16": int16,
    "int32": int32,
    "int64": int64,
    "uint8": uint8,
    "uint16": uint16,
    "uint32": uint32,
    "uint64": uint64,
    "float16": float16,
    "float32": float32,
    "float64": float64,
    "complex64": complex64,
    "complex128": complex128,
}


def _validate_device(device: Any) -> None:
    normalize_device(device)


class ArrayNamespaceInfo:
    def capabilities(self) -> dict[str, builtins.bool | builtins.int | None]:
        return {
            "boolean indexing": False,
            "data-dependent shapes": False,
            "max dimensions": 5,
        }

    def default_device(self):
        return serial()

    def devices(self) -> list[AbstractDevice]:
        return [serial(), cpu()]

    def default_dtypes(self, *, device=None) -> dict[str, Any]:
        _validate_device(device)
        return {
            "real floating": float,
            "complex floating": complex128,
            "integral": int,
            "indexing": intp,
        }

    def dtypes(self, *, device=None, kind=None) -> dict[str, Any]:
        _validate_device(device)
        if kind is None:
            return dict(_DTYPES)
        return {name: dtype for name, dtype in _DTYPES.items() if isdtype(dtype, kind)}


def __array_namespace_info__() -> ArrayNamespaceInfo:
    return ArrayNamespaceInfo()


__all__ = ["ArrayNamespaceInfo", "__array_namespace_info__"]
