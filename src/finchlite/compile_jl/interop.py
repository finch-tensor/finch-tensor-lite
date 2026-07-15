from __future__ import annotations

from typing import Any, cast

import numpy as np

from finchlite.algebra import TupleFType
from finchlite.codegen import NumpyBuffer
from finchlite.finch_assembly import Buffer
from finchlite.tensor import (
    BufferizedNDArray,
    DenseLevel,
    ElementLevel,
    FiberTensor,
    Level,
    Scalar,
    SparseByteMapLevel,
    SparseCOOLevel,
    SparseHashLevel,
    SparseListLevel,
)
from finchlite.tensor.np_wrapper import NumPyWrapper

from . import dtypes as jl_dtypes
from .julia import jc, jl


def is_julia_obj(obj: Any) -> bool:
    return isinstance(obj, jc.AnyValue)


def _as_julia_scalar(val):
    if isinstance(val, np.bool_):
        return val.item()
    return val


def _buffer_to_jl(buffer: Buffer):
    if isinstance(buffer, NumpyBuffer):
        if isinstance(buffer.ftype.element_type, TupleFType):
            data = jl.PythonCall.PyArray(jl.PythonCall.Py(buffer.arr))
            tuple_type = jl.Tuple[jl.fieldtypes(jl.eltype(data))]
            return jl.reinterpret(tuple_type, data)
        return jl.Vector(buffer.arr)
    raise ValueError(f"Unsupported buffer type: {type(buffer)}")


def _plus_one_buffer_to_jl(buffer: Buffer):
    return jl.Finch.PlusOneVector(_buffer_to_jl(buffer))


def level_to_jl(level: Level):
    match level:
        case ElementLevel():
            return jl.ElementLevel(
                _as_julia_scalar(level.fill_value),
                _buffer_to_jl(level.val),
            )
        case DenseLevel(lvl=lvl, dimension=dimension):
            return jl.DenseLevel(level_to_jl(lvl), int(dimension))
        case SparseListLevel(lvl=lvl, dimension=dimension, ptr=ptr, idx=idx):
            if ptr is None or idx is None:
                raise ValueError("SparseListLevel must have ptr and idx buffers")
            return jl.SparseListLevel(
                level_to_jl(lvl),
                int(dimension),
                _plus_one_buffer_to_jl(cast(Buffer, ptr)),
                _plus_one_buffer_to_jl(cast(Buffer, idx)),
            )
        case SparseByteMapLevel(
            lvl=lvl, dimension=dimension, ptr=ptr, tbl=tbl, srt=srt
        ):
            if ptr is None or tbl is None or srt is None:
                raise ValueError(
                    "SparseByteMapLevel must have ptr, tbl, and srt buffers"
                )
            return jl.SparseByteMapLevel(
                level_to_jl(lvl),
                int(dimension),
                _plus_one_buffer_to_jl(cast(Buffer, ptr)),
                _buffer_to_jl(cast(Buffer, tbl)),
                _plus_one_buffer_to_jl(cast(Buffer, srt)),
            )
        case SparseCOOLevel(lvl=lvl, coo_shape=coo_shape, ptr=ptr, tbl=tbl):
            return jl.SparseCOOLevel(
                level_to_jl(lvl),
                tuple(int(dim) for dim in coo_shape),
                _plus_one_buffer_to_jl(ptr),
                tuple(_plus_one_buffer_to_jl(idx) for idx in tbl),
            )
        case SparseHashLevel(
            lvl=lvl,
            dimension=dimension,
            ptr=ptr,
            tbl_ctrl=tbl_ctrl,
            tbl=tbl,
            pool=pool,
            perm=perm,
            subtables=subtables,
            single_writer=single_writer,
        ):
            if (
                ptr is None
                or tbl_ctrl is None
                or tbl is None
                or pool is None
                or perm is None
            ):
                raise ValueError(
                    "SparseHashLevel must have ptr, tbl_ctrl, tbl, pool, and perm "
                    "buffers"
                )
            dimension = _as_julia_scalar(np.asarray(dimension).item())
            constructor = jl.SparseHashLevel[(jl.typeof(dimension), single_writer)]
            return constructor(
                level_to_jl(lvl),
                dimension,
                int(subtables),
                _plus_one_buffer_to_jl(cast(Buffer, ptr)),
                _buffer_to_jl(cast(Buffer, tbl_ctrl)),
                _plus_one_buffer_to_jl(cast(Buffer, tbl)),
                _plus_one_buffer_to_jl(cast(Buffer, pool)),
                _plus_one_buffer_to_jl(cast(Buffer, perm)),
            )
        case _:
            raise ValueError(f"Unsupported Finch level type: {type(level)}")


def _ndarray_to_jl_tensor(
    arr: np.ndarray,
    fill_value: Any,
    *,
    copy: bool = False,
):
    if copy:
        arr = arr.copy() if arr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(arr)
    elif not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    buf = jl.Vector(np.reshape(arr, -1))
    fill = _as_julia_scalar(np.asarray(fill_value, dtype=arr.dtype)[()])
    lvl = jl.ElementLevel(fill, buf)
    for dim in reversed(arr.shape):
        lvl = jl.DenseLevel(lvl, int(dim))
    return jl.Tensor(lvl)


def tensor_to_jl(obj):
    if is_julia_obj(obj) and jl.isa(obj, jl.Finch.Tensor):
        return obj
    if isinstance(obj, FiberTensor):
        if obj.pos != 0:
            raise ValueError("Only root-position FiberTensor objects can use Julia")
        return jl.Tensor(level_to_jl(obj.lvl))
    if isinstance(obj, BufferizedNDArray):
        return _ndarray_to_jl_tensor(obj.to_numpy(), obj.fill_value, copy=False)
    if isinstance(obj, NumPyWrapper):
        return _ndarray_to_jl_tensor(obj._data, obj.fill_value, copy=False)
    if isinstance(obj, Scalar):
        return scalar_to_jl(obj.val)
    if isinstance(obj, np.ndarray):
        fill = np.asarray(0, dtype=obj.dtype)[()]
        return _ndarray_to_jl_tensor(obj, fill, copy=False)
    if isinstance(obj, np.generic):
        return scalar_to_jl(obj.item())
    if np.isscalar(obj):
        return scalar_to_jl(obj)
    if hasattr(obj, "val"):
        return scalar_to_jl(obj.val)
    raise ValueError(f"Unsupported Julia backend argument type: {type(obj)}")


def scalar_to_jl(val):
    if isinstance(val, np.generic):
        val = val.item()
    buf = np.asarray([val])
    return jl.Tensor(jl.ElementLevel(_as_julia_scalar(buf.item()), jl.Vector(buf)))


def _dense_jl_tensor_to_numpy(obj) -> np.ndarray:
    if len(tuple(jl.size(obj))) == 0:
        return np.asarray(jl.getindex(obj))

    if _is_dense_jl_tensor(obj):
        shape = tuple(jl.size(obj))
        dense_level = obj.lvl
    else:
        shape = tuple(jl.size(obj))
        fill_value = jl.fill_value(obj)
        fill_type = jl_dtypes.to_jl_type(jl_dtypes.to_fl_dtype(fill_value))
        dense_level = jl.Element(jc.convert(fill_type, fill_value))
        for _ in shape:
            dense_level = jl.Dense(dense_level)
        dense_level = jl.Tensor(dense_level, obj).lvl

    for _ in shape:
        dense_level = dense_level.lvl

    arr = np.asarray(jl.reshape(dense_level.val, shape))
    if len(shape) > 0:
        arr = arr.transpose()
    return arr.copy()


def _is_dense_jl_tensor(obj) -> bool:
    lvl = obj.lvl
    for _ in tuple(jl.size(obj)):
        if not jl.isa(lvl, jl.Finch.Dense):
            return False
        lvl = lvl.lvl
    return True


def jl_tensor_to_python(obj):
    if not (is_julia_obj(obj) and jl.isa(obj, jl.Finch.Tensor)):
        return obj
    return BufferizedNDArray.from_numpy(
        _dense_jl_tensor_to_numpy(obj),
        fill_value=jl.fill_value(obj),
    )
