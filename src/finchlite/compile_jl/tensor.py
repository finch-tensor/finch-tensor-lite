import operator
from typing import Any

import numpy as np

from finchlite import Tensor, TensorFType
from finchlite.tensor.override_tensor import OverrideTensor

from . import dtypes as jl_dtypes
from .julia import jc, jl
from .levels import (
    ElementFormat,
    LevelFormat,
    SparseCOOFormat,
    jlobj_to_format,
)
from .typing import DType, JLFType, JuliaObj, number
from .utils import add_missing_dims, add_plus_one, expand_ellipsis


# Tensor Class and associated ftype
class FinchJLTensorFType(TensorFType, JLFType):
    def __init__(self, lvl):
        self._lvl: LevelFormat = lvl

    @property
    def ndim(self) -> np.intp:
        return self._lvl.ndim

    @property
    def fill_value(self) -> Any:
        return self._lvl.fill_value

    @property
    def element_type(self) -> Any:
        return self._lvl.element_type

    @property
    def dtype(self) -> Any:
        return self.element_type

    @property
    def shape_type(self) -> tuple:
        return tuple(reversed(self._lvl.shape_type))

    @property
    def jl_type(self):
        return jl.Finch.Tensor[self.format.jl_type]

    def construct(self, shape: tuple) -> Tensor:
        # EXPERIMENTAL reversed-axis convention: jl.size is always kept as
        # the reverse of the Python-facing shape; FinchJLTensor.shape un-
        # reverses it back on the way out (see there for the full rationale).
        return FinchJLTensor(
            jl.Finch.Tensor(self._lvl.create_jl_obj(), tuple(reversed(shape)))
        )

    def from_numpy(self, _) -> Tensor:
        raise NotImplementedError

    def __call__(self, val: Any) -> Tensor:
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )

    def __eq__(self, other):
        if not isinstance(other, FinchJLTensorFType):
            return False
        return self._lvl == other._lvl

    def __hash__(self):
        return hash(("FinchJLTensorFType", self._lvl))


class FinchJLTensor(OverrideTensor):
    def override_module(self):
        import finch

        return finch

    def __array_function__(self, func, types, args, kwargs):
        # Guard np.asarray specifically: lazy.asarray() calls np.asarray()
        # internally, and redirecting that back to finch.asarray() creates an
        # infinite loop (finch.asarray returns FinchJLTensor unchanged, then
        # lazy.asarray calls np.asarray again, etc.). Returning NotImplemented
        # lets numpy fall back to the dtype=object path in lazy.asarray, which
        # correctly returns the tensor as-is for downstream Julia compilation.
        import finch

        if func.__name__ == "asarray":
            return NotImplemented
        override_func = getattr(finch, func.__name__, None)
        if override_func is None:
            return NotImplemented
        return override_func(*args, **kwargs)

    def __init__(self, obj: JuliaObj):
        if isinstance(obj, JuliaObj):
            assert jl.isa(obj, jl.Finch.Tensor)
            self._obj = obj
        else:
            raise ValueError(f"Raw julia object expected. Found: {type(obj)}")

    @property
    def ftype(self) -> TensorFType:
        """Returns the ftype of the buffer"""
        return FinchJLTensorFType(jlobj_to_format(self._obj.lvl))

    @property
    def dtype(self) -> Any:
        return self.element_type

    @property
    def shape(self) -> tuple:
        """Shape of the tensor.

        EXPERIMENTAL: jl.size is always kept as the reverse of the
        Python-facing shape (see asarray/full/__getitem__/todense), so this
        un-reverses it back to Python axis order.
        """
        return tuple(reversed(jl.size(self._obj)))

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Array API behavior: indexing a 0-D array with () is a no-op.
        if self.shape == () and key == ():
            return self

        # standard indexing mode
        key = expand_ellipsis(key, self.shape)
        key = add_missing_dims(key, self.shape)
        key = add_plus_one(key, self.shape)

        if all(isinstance(k, int) for k in key):
            return jl.getindex(self._obj, *reversed(key))

        # Finch's getindex has no notion of `None`/newaxis, so strip those
        # entries out, index normally, then re-insert size-1 axes into the
        # result at the positions implied by the original key.
        real_key = tuple(k for k in key if k is not jl.nothing)
        newaxis_positions = []
        axis = 0
        for k in key:
            if k is jl.nothing:
                newaxis_positions.append(axis)
                axis += 1
            elif not isinstance(k, int):
                axis += 1

        result = jl.getindex(self._obj, *reversed(real_key))

        if not newaxis_positions:
            assert jl.isa(result, jl.Finch.Tensor)
            return FinchJLTensor(result)

        arr = (
            FinchJLTensor(result).todense()
            if jl.isa(result, jl.Finch.Tensor)
            else np.asarray(result)
        )
        for pos in newaxis_positions:
            arr = np.expand_dims(arr, pos)
        return asarray(arr)

    def _is_dense(self) -> bool:
        lvl = self._obj.lvl
        for _ in self.shape:
            if not jl.isa(lvl, jl.Finch.Dense):
                return False
            lvl = lvl.lvl
        return True

    def todense(self) -> np.ndarray:
        obj = self._obj

        if self.ndim == 0:  # early return for 0-D tensor.
            return np.array(jl.fill_value(obj))

        if self._is_dense():
            # don't materialize a dense finch tensor
            shape = jl.size(obj)
            dense_tensor = obj.lvl
        else:
            # create materialized dense array
            shape = jl.size(obj)
            dense_lvls = jl.Element(
                jc.convert(jl_dtypes.fl_dtype_to_jl[self.dtype], jl.fill_value(obj))
            )
            for _ in range(self.ndim):
                dense_lvls = jl.Dense(dense_lvls)
            dense_tensor = jl.Tensor(dense_lvls, obj).lvl  # materialize

        for _ in range(self.ndim):
            dense_tensor = dense_tensor.lvl

        # `shape` here is jl.size(obj), i.e. the reversed-axis shape; reshape
        # into that, then transpose (a cheap stride-only view) back to the
        # Python-facing axis order.
        arr = jl.reshape(dense_tensor.val, tuple(shape))
        return np.asarray(arr).transpose()

    def __eq__(self, other):
        return isinstance(other, FinchJLTensor) and self._obj == other._obj

    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        # A 0-d tensor has no axes to permute, and an empty swizzle
        # permutation crashes Finch's SwizzleArray size/show machinery.
        if self.ndim == 0:
            return jl.sprint(jl.show, jl.MIME("text/plain"), self._obj)
        swiz = jl.swizzle(self._obj, tuple(reversed(range(self.ndim, 1, -1))))
        return jl.sprint(jl.show, jl.MIME("text/plain"), swiz)

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        if api_version is None:
            api_version = "2024.12"

        if api_version not in {"2021.12", "2022.12", "2023.12", "2024.12"}:
            raise ValueError(f'"{api_version}" Array API version not supported.')
        import finch

        return finch

    def copy(self) -> "FinchJLTensor":
        return FinchJLTensor(jl.deepcopy(self._obj))

    def _scalar_value(self):
        if self.shape != ():
            raise TypeError(
                "only 0-dimensional arrays can be converted to Python scalars"
            )
        return jl.getindex(self._obj)

    def __bool__(self) -> bool:
        return bool(self._scalar_value())

    def __int__(self) -> int:
        return int(self._scalar_value())

    def __float__(self) -> float:
        return float(self._scalar_value())

    def __index__(self) -> int:
        try:
            return operator.index(self._scalar_value())
        except TypeError as exc:
            raise TypeError(
                "only integer scalar arrays can be converted to an index"
            ) from exc


def asarray(
    obj,
    /,
    *,
    dtype: DType | None = None,
    fill_value: np.number | None = None,
    copy: bool | None = None,
) -> FinchJLTensor:
    if fill_value is None:
        fill_value = 0.0
    if isinstance(obj, FinchJLTensor):
        if copy:
            return obj.copy()
        return obj
    if copy is None:
        copy = True
    if isinstance(obj, int | float | complex | bool | list):
        if copy is False:
            raise ValueError(
                "copy=False isn't supported for scalar inputs and Python lists"
            )
        obj = np.asarray(obj)
    if isinstance(obj, np.ndarray):
        if dtype is not None:
            obj = np.asarray(obj, dtype=jl_dtypes.jl_to_np_dtype[dtype])

        if obj.ndim == 0:
            return full((), obj.item(), dtype=dtype)

        # EXPERIMENTAL reversed-axis convention: keep the buffer in its
        # natural C (row-major) layout -- which is exactly the column-major
        # layout of the *reversed*-shape tensor -- instead of permuting data
        # to Fortran order, and build the DenseLevel nest in reverse axis
        # order so jl.size ends up as reversed(obj.shape).
        if copy:
            obj = obj.copy() if obj.flags["C_CONTIGUOUS"] else np.ascontiguousarray(obj)
        else:
            if not obj.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    "Unable to avoid copy while creating an array as requested."
                )
        buf = np.reshape(obj, -1)

        lvl = jl.ElementLevel(np.asarray(fill_value, dtype=obj.dtype).item(), buf)
        for i in reversed(obj.shape):
            lvl = jl.DenseLevel(lvl, i)
        return FinchJLTensor(jl.Tensor(lvl))
    if hasattr(obj, "__module__") and obj.__module__.startswith("scipy.sparse"):
        if copy:
            if obj.format == "csr":
                obj = obj.copy() if obj.has_sorted_indices else obj.sorted_indices()
                if not obj.has_canonical_format:
                    obj.sum_duplicates()
            elif obj.format == "coo":
                obj = obj.copy()
                obj.sum_duplicates()
            else:
                obj = obj.asformat("csr")
        if (
            copy is False
            and obj.format not in ("coo", "csr")
            and not obj.has_canonical_format
        ):
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )
        m, n = obj.shape
        if dtype is not None:
            fill_value = np.asarray(
                fill_value, dtype=jl_dtypes.jl_to_np_dtype[dtype]
            ).item()
        if obj.format == "coo":
            # EXPERIMENTAL reversed-axis convention: shape and coordinate
            # order are both given reversed ((n, m) and (col, row)), so axis
            # 0 of the SparseCOOLevel is "col". Sorting must vary axis 0
            # (col) fastest -- i.e. row primary, col secondary -- so col is
            # lexsort's secondary (first) key and row its primary (last) key.
            order = np.lexsort((obj.col, obj.row))
            row_s = obj.row[order]
            col_s = obj.col[order]
            data_s = obj.data[order]
            nnz = len(data_s)
            return FinchJLTensor(
                jl.Tensor(
                    jl.SparseCOOLevel(
                        jl.ElementLevel(fill_value, data_s),
                        (n, m),
                        # ptr marks the single coordinate block [1, nnz+1);
                        # it's a plain Python list, so it needs an explicit
                        # jl.Vector to become a real Julia array (numpy arrays
                        # get this automatically via PythonCall's zero-copy
                        # PyArray wrapping).
                        jl.Vector([1, nnz + 1]),
                        (
                            jl.Finch.PlusOneVector(col_s),
                            jl.Finch.PlusOneVector(row_s),
                        ),
                    )
                )
            )
        if obj.format == "csr":
            return FinchJLTensor(
                jl.Tensor(
                    jl.DenseLevel(
                        jl.SparseListLevel(
                            jl.ElementLevel(fill_value, obj.data),
                            m,
                            jl.Finch.PlusOneVector(obj.indptr),
                            jl.Finch.PlusOneVector(obj.indices),
                        ),
                        n,
                    )
                )
            )
        raise ValueError(f"Unsupported SciPy format: {type(obj)}")
    raise ValueError(
        f"Either numpy array or a Finch tensor should be provided. Found: {type(obj)}"
    )


def reshape(
    x: FinchJLTensor, /, shape: tuple[int, ...], *, copy: bool | None = None
) -> FinchJLTensor:
    if copy is False:
        raise ValueError("Unable to avoid copy during reshape.")
    if all(i == 1 for i in x.shape):
        return full(shape, x[tuple(i - 1 for i in x.shape)], dtype=x.dtype)
    return FinchJLTensor(jl.reshape(x._obj, tuple(reversed(shape))))


def full(
    shape: int | tuple[int, ...],
    val: number,
    *,
    dtype: DType | None = None,
    format=None,
) -> FinchJLTensor:
    if not np.isscalar(val):
        raise ValueError("`fill_value` must be a scalar")
    if isinstance(shape, int):
        shape = (shape,)
    dtype = (
        np.asarray(val).dtype.type if dtype is None else jl_dtypes.jl_to_np_dtype[dtype]
    )
    if dtype == np.bool_:  # Fails with: Finch currently only supports isbits defaults
        dtype = bool

    # Rank-0 tensors should be represented as a leaf element level.
    # Building them through SparseCOO requires an explicit rank parameter.
    if len(shape) == 0 and format is None:
        elt_fmt = ElementFormat(val, dtype)
        cast_val = elt_fmt.fill_value
        if isinstance(cast_val, np.generic):
            cast_val = cast_val.item()
        # A bare jl.Element(val) treats `val` as the level's default and
        # allocates an empty buffer, so reading it back gives the type-zero,
        # not `val`. A rank-0 tensor needs an explicit length-1 buffer.
        buf = np.asarray([cast_val], dtype=dtype)
        return FinchJLTensor(jl.Tensor(jl.ElementLevel(cast_val, buf)))

    if format is None:
        format = SparseCOOFormat(ElementFormat(val, dtype), len(shape))

    if format.fill_value != val:
        # jl.Tensor(lvl, real_array) infers jl.size literally from the
        # array's own .shape (no implicit reversal on that path -- proven
        # separately), so under the reversed-axis convention the array
        # passed here must itself already be shaped in reverse.
        return FinchJLTensor(
            jl.Tensor(
                format.create_jl_obj(),
                np.full(tuple(reversed(shape)), val, dtype=dtype),
            )
        )
    return FinchJLTensor(jl.Tensor(format.create_jl_obj(), *reversed(shape)))


def full_like(
    x: FinchJLTensor,
    /,
    fill_value: number,
    *,
    dtype: DType | None = None,
    format=None,
) -> FinchJLTensor:
    return full(x.shape, fill_value, dtype=dtype, format=format)


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    format=None,
) -> FinchJLTensor:
    return full(shape, np.float64(1), dtype=dtype, format=format)


def ones_like(
    x: FinchJLTensor,
    /,
    *,
    dtype: DType | None = None,
    format=None,
) -> FinchJLTensor:
    dtype = x.dtype if dtype is None else dtype
    return ones(x.shape, dtype=dtype, format=format)


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    format=None,
) -> FinchJLTensor:
    return full(shape, np.float64(0), dtype=dtype, format=format)


def zeros_like(
    x: FinchJLTensor,
    /,
    *,
    dtype: DType | None = None,
    format=None,
) -> FinchJLTensor:
    dtype = x.dtype if dtype is None else dtype
    return zeros(x.shape, dtype=dtype, format=format)


def empty(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    format=None,
) -> FinchJLTensor:
    return full(shape, np.float64(0), dtype=dtype, format=format)


def empty_like(
    x: FinchJLTensor,
    /,
    *,
    dtype: DType | None = None,
    format=None,
) -> FinchJLTensor:
    dtype = x.dtype if dtype is None else dtype
    return empty(x.shape, dtype=dtype, format=format)


def arange(
    start: float,
    /,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: DType | None = None,
) -> FinchJLTensor:
    return asarray(np.arange(start, stop, step, jl_dtypes.jl_to_np_dtype[dtype]))


def real(  # finchlite versions caused infinite recursion.
    x: FinchJLTensor,
    /,
    *,
    dtype: DType | None = None,
) -> FinchJLTensor:
    return asarray(np.real(x.todense()), dtype=jl_dtypes.jl_to_np_dtype[dtype])


def imag(  # finchlite versions caused infinite recursion.
    x: FinchJLTensor,
    /,
    *,
    dtype: DType | None = None,
) -> FinchJLTensor:
    return asarray(np.imag(x.todense()), dtype=jl_dtypes.jl_to_np_dtype[dtype])


def _to_numpy(x):
    if isinstance(x, FinchJLTensor):
        return x.todense()
    return np.asarray(x)


def where(
    condition,
    x1,
    x2,
    /,
) -> FinchJLTensor:
    return asarray(np.where(_to_numpy(condition), _to_numpy(x1), _to_numpy(x2)))


def linspace(
    start: complex,
    stop: complex,
    /,
    num: int,
    *,
    dtype: DType | None = None,
    endpoint: bool = True,
) -> FinchJLTensor:
    return asarray(
        np.linspace(
            start,
            stop,
            num=num,
            dtype=jl_dtypes.jl_to_np_dtype[dtype],
            endpoint=endpoint,
        )
    )
