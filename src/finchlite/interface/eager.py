import builtins
import functools
import operator
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.fft as scipy_fft
import scipy.linalg as scipy_linalg
import scipy.sparse.linalg as scipy_sparse_linalg

from finchlite.algebra import FinchOperator, to_numpy, to_scipy

from . import lazy
from .fuse import compute


def _warn_compute(x, op_name: str):
    if isinstance(x, lazy.LazyTensor):
        warnings.warn(
            f"{op_name} requires a materialized array; computing lazy operand.",
            RuntimeWarning,
            stacklevel=3,
        )
        return compute(x)
    return x


def full(
    shape: int | tuple[int, ...],
    fill_value: bool | complex,
    *,
    dtype: Any | None = None,
    device=None,
):
    return compute(lazy.full(shape, fill_value, dtype=dtype, device=device))


def zeros(shape: int | tuple[int, ...], *, dtype: Any | None = None, device=None):
    return compute(lazy.zeros(shape, dtype=dtype, device=device))


def ones(shape: int | tuple[int, ...], *, dtype: Any | None = None, device=None):
    return compute(lazy.ones(shape, dtype=dtype, device=device))


def empty(shape: int | tuple[int, ...], *, dtype: Any | None = None, device=None):
    return compute(lazy.empty(shape, dtype=dtype, device=device))


def eye(
    n_rows: int,
    n_cols: int | None = None,
    *,
    k: int = 0,
    dtype: Any | None = None,
    device=None,
):
    return compute(lazy.eye(n_rows, n_cols, k=k, dtype=dtype, device=device))


def triu(x, /, *, k: int = 0):
    if isinstance(x, lazy.LazyTensor):
        return lazy.triu(x, k=k)
    return compute(lazy.triu(x, k=k))


def tril(x, /, *, k: int = 0):
    if isinstance(x, lazy.LazyTensor):
        return lazy.tril(x, k=k)
    return compute(lazy.tril(x, k=k))


def diag(x, /, *, k: int = 0):
    if isinstance(x, lazy.LazyTensor):
        return lazy.diag(x, k=k)
    return compute(lazy.diag(x, k=k))


def diff(x, /, *, axis: int = -1, n: int = 1, prepend=None, append=None):
    if (
        isinstance(x, lazy.LazyTensor)
        or isinstance(prepend, lazy.LazyTensor)
        or isinstance(append, lazy.LazyTensor)
    ):
        return lazy.diff(x, axis=axis, n=n, prepend=prepend, append=append)
    return compute(lazy.diff(x, axis=axis, n=n, prepend=prepend, append=append))


def cumulative_sum(
    x,
    /,
    *,
    axis: int | None = None,
    dtype=None,
    include_initial: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.cumulative_sum(
            x,
            axis=axis,
            dtype=dtype,
            include_initial=include_initial,
        )
    return compute(
        lazy.cumulative_sum(
            x,
            axis=axis,
            dtype=dtype,
            include_initial=include_initial,
        )
    )


def cumulative_prod(
    x,
    /,
    *,
    axis: int | None = None,
    dtype=None,
    include_initial: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.cumulative_prod(
            x,
            axis=axis,
            dtype=dtype,
            include_initial=include_initial,
        )
    return compute(
        lazy.cumulative_prod(
            x,
            axis=axis,
            dtype=dtype,
            include_initial=include_initial,
        )
    )


def diagonal(x, /, *, offset: int = 0):
    if isinstance(x, lazy.LazyTensor):
        return lazy.diagonal(x, offset=offset)
    return compute(lazy.diagonal(x, offset=offset))


def trace(x, /, *, offset: int = 0, dtype=None):
    if isinstance(x, lazy.LazyTensor):
        return lazy.trace(x, offset=offset, dtype=dtype)
    return compute(lazy.trace(x, offset=offset, dtype=dtype))


def full_like(
    x, /, fill_value: bool | complex, *, dtype: Any | None = None, device=None
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.full_like(x, fill_value, dtype=dtype, device=device)
    return compute(lazy.full_like(x, fill_value, dtype=dtype, device=device))


def empty_like(x, /, *, dtype: Any | None = None, device=None):
    if isinstance(x, lazy.LazyTensor):
        return lazy.empty_like(x, dtype=dtype, device=device)
    return compute(lazy.empty_like(x, dtype=dtype, device=device))


def zeros_like(x, /, *, dtype: Any | None = None, device=None):
    if isinstance(x, lazy.LazyTensor):
        return lazy.zeros_like(x, dtype=dtype, device=device)
    return full_like(x, 0, dtype=dtype, device=device)


def ones_like(x, /, *, dtype: Any | None = None, device=None):
    if isinstance(x, lazy.LazyTensor):
        return lazy.ones_like(x, dtype=dtype, device=device)
    return full_like(x, 1, dtype=dtype, device=device)


def arange(
    start: float,
    /,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: Any | None = None,
    device=None,
):
    return compute(lazy.arange(start, stop, step, dtype=dtype, device=device))


def linspace(
    start: float,
    stop: float,
    /,
    num: int,
    *,
    dtype: Any | None = None,
    device=None,
    endpoint: bool = True,
):
    return compute(
        lazy.linspace(start, stop, num, dtype=dtype, endpoint=endpoint, device=device)
    )


def permute_dims(arg, /, axes: tuple[int, ...]):
    if isinstance(arg, lazy.LazyTensor):
        return lazy.permute_dims(arg, axes=axes)
    return compute(lazy.permute_dims(arg, axes=axes))


def expand_dims(
    x,
    /,
    axis: int | tuple[int, ...] = 0,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.expand_dims(x, axis=axis)
    return compute(lazy.expand_dims(x, axis=axis))


def squeeze(
    x,
    /,
    axis: int | tuple[int, ...],
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.squeeze(x, axis=axis)
    return compute(lazy.squeeze(x, axis=axis))


def astype(x, dtype, /, *, copy=True, device=None):
    if isinstance(x, lazy.LazyTensor):
        return lazy.astype(x, dtype, copy=copy, device=device)
    return compute(lazy.astype(x, dtype, copy=copy, device=device))


def reduce(
    op: FinchOperator,
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype=None,
    keepdims: bool = False,
    init=None,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.reduce(op, x, axis=axis, dtype=dtype, keepdims=keepdims, init=init)
    return compute(
        lazy.reduce(op, x, axis=axis, dtype=dtype, keepdims=keepdims, init=init)
    )


def round(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.round(x)
    return compute(lazy.round(x))


def floor(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.floor(x)
    return compute(lazy.floor(x))


def ceil(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.ceil(x)
    return compute(lazy.ceil(x))


def trunc(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.trunc(x)
    return compute(lazy.trunc(x))


def sum(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype=None,
    keepdims: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
    return compute(lazy.sum(x, axis=axis, dtype=dtype, keepdims=keepdims))


def prod(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype=None,
    keepdims: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)
    return compute(lazy.prod(x, axis=axis, dtype=dtype, keepdims=keepdims))


def argmin(
    x,
    /,
    *,
    axis: int | None = None,
    keepdims: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.argmin(x, axis=axis, keepdims=keepdims)
    return compute(lazy.argmin(x, axis=axis, keepdims=keepdims))


def argmax(
    x,
    /,
    *,
    axis: int | None = None,
    keepdims: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.argmax(x, axis=axis, keepdims=keepdims)
    return compute(lazy.argmax(x, axis=axis, keepdims=keepdims))


def elementwise(f: FinchOperator, *args):
    if builtins.any(isinstance(arg, lazy.LazyTensor) for arg in args):
        return lazy.elementwise(f, *args)
    return compute(lazy.elementwise(f, *args))


def add(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.add(x1, x2)
    return compute(lazy.add(x1, x2))


def reciprocal(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.reciprocal(x)
    return compute(lazy.reciprocal(x))


def subtract(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.subtract(x1, x2)
    return compute(lazy.subtract(x1, x2))


def multiply(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.multiply(x1, x2)
    return compute(lazy.multiply(x1, x2))


def outer(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.outer(x1, x2)
    return compute(lazy.outer(x1, x2))


def divide(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.divide(x1, x2)
    return compute(lazy.divide(x1, x2))


def abs(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.abs(x)
    return compute(lazy.abs(x))


def positive(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.positive(x)
    return compute(lazy.positive(x))


def negative(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.negative(x)
    return compute(lazy.negative(x))


def matmul(x1, x2, /):
    """
    Computes the matrix product.

    Returns a LazyTensor if either x1 or x2 is a LazyTensor.
    Otherwise, computes the result eagerly.
    """
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.matmul(x1, x2)
    c = lazy.matmul(x1, x2)
    return compute(c)


def matrix_transpose(x, /):
    """
    Computes the transpose of a matrix or stack of matrices.
    """
    if isinstance(x, lazy.LazyTensor):
        return lazy.matrix_transpose(x)
    return compute(lazy.matrix_transpose(x))


def inv(x, /):
    x = _warn_compute(x, "inv")
    try:
        return lazy.asarray(
            to_numpy(scipy_sparse_linalg.inv(to_scipy(lazy.asarray(x))))
        )
    except Exception:
        pass
    x = to_numpy(lazy.asarray(x))
    return lazy.asarray(np.ascontiguousarray(np.linalg.inv(x)))


def cholesky(x, /, *, upper=False):
    x = _warn_compute(x, "cholesky")
    x = to_numpy(lazy.asarray(x))
    result = np.linalg.cholesky(x)
    if upper:
        result = np.swapaxes(np.conjugate(result), -1, -2)
    return lazy.asarray(result)


def cross(x1, x2, /, *, axis=-1):
    x1 = _warn_compute(x1, "cross")
    x1 = to_numpy(lazy.asarray(x1))
    x2 = _warn_compute(x2, "cross")
    x2 = to_numpy(lazy.asarray(x2))
    cross_func = getattr(np.linalg, "cross", np.cross)
    return lazy.asarray(cross_func(x1, x2, axis=axis))

def minimumSwaps(arr): 
    """
    Minimum number of swaps needed to order a
    permutation array
    """
    # from https://www.thepoorcoder.com/hackerrank-minimum-swaps-2-solution/
    a = dict(enumerate(arr))
    b = {v:k for k,v in a.items()}
    count = 0
    for i in a:
        x = a[i]
        if x!=i:
            y = b[i]
            a[y] = x
            b[x] = y
            count+=1
    return count

def det(x, /):
    x = _warn_compute(x, "det")
    try:
        x_sp = to_scipy(lazy.asarray(x))
        lu = scipy_sparse_linalg.splu(x_sp.tocsc())
        diag_u = lu.U.diagonal()

        perm_sign = 1
        for perm in (lu.perm_r, lu.perm_c):
            seen = np.zeros(perm.size, dtype=bool)
            swaps = 0
            for i in range(perm.size):
                if seen[i]:
                    continue
                j = i
                cycle_len = 0
                while not seen[j]:
                    seen[j] = True
                    j = perm[j]
                    cycle_len += 1
                swaps += builtins.max(cycle_len - 1, 0)
            if swaps % 2:
                perm_sign *= -1

        return lazy.asarray(np.asarray(perm_sign * diag_u.prod()))
    except Exception:
        pass
    x = to_numpy(lazy.asarray(x))
    return lazy.asarray(np.asarray(np.linalg.det(x)))


def lu(x, /, *, permute_l=False, p_indices=False):
    x = _warn_compute(x, "lu")
    try:
        return scipy_sparse_linalg.splu(to_scipy(lazy.asarray(x)).tocsc())
    except Exception:
        pass
    x = to_numpy(lazy.asarray(x))
    return tuple(
        lazy.asarray(part)
        for part in scipy_linalg.lu(x, permute_l=permute_l, p_indices=p_indices)
    )


def eigh(x, /, *, k=None, rtol=None, atol=None):
    x = _warn_compute(x, "eigh")
    if k is not None:
        try:
            kwargs = {
                "k": k,
                "return_eigenvectors": True,
            }
            if rtol is not None and atol is not None:
                warnings.warn(
                    "eigh sparse fallback supports only one tolerance; "
                    "using min(rtol, atol).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                tol = builtins.min(rtol, atol)
            elif rtol is not None:
                tol = rtol
            else:
                tol = atol
            if tol is not None:
                kwargs["tol"] = tol
            return lazy.asarray(
                scipy_sparse_linalg.eigsh(to_scipy(lazy.asarray(x)), **kwargs)
            )
        except Exception:
            pass
    if rtol is not None or atol is not None:
        warnings.warn(
            "eigh dense fallback does not support rtol or atol; "
            "ignoring tolerance arguments.",
            RuntimeWarning,
            stacklevel=2,
        )
    x = to_numpy(lazy.asarray(x))
    return lazy.asarray(np.linalg.eigh(x))


def eigvalsh(x, /, *, k=None, rtol=None, atol=None):
    x = _warn_compute(x, "eigvalsh")
    if k is not None:
        try:
            kwargs = {
                "k": k,
                "return_eigenvectors": False,
            }
            if rtol is not None and atol is not None:
                warnings.warn(
                    "eigvalsh sparse fallback supports only one tolerance; "
                    "using min(rtol, atol).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                tol = builtins.min(rtol, atol)
            elif rtol is not None:
                tol = rtol
            else:
                tol = atol
            if tol is not None:
                kwargs["tol"] = tol
            return lazy.asarray(
                scipy_sparse_linalg.eigsh(to_scipy(lazy.asarray(x)), **kwargs)
            )
        except Exception:
            pass
    if rtol is not None or atol is not None:
        warnings.warn(
            "eigvalsh dense fallback does not support rtol or atol; "
            "ignoring tolerance arguments.",
            RuntimeWarning,
            stacklevel=2,
        )
    x = to_numpy(lazy.asarray(x))
    return lazy.asarray(np.linalg.eigvalsh(x))


def matrix_rank(x, /, *, rtol=None, atol=None):
    x = _warn_compute(x, "matrix_rank")
    x = to_numpy(lazy.asarray(x))
    if atol is not None:
        if rtol is not None:
            warnings.warn(
                "matrix_rank cannot apply both rtol and atol with this fallback; "
                "using atol.",
                RuntimeWarning,
                stacklevel=2,
            )
        return lazy.asarray(np.asarray(np.linalg.matrix_rank(x, tol=atol)))
    if rtol is None:
        return lazy.asarray(np.asarray(np.linalg.matrix_rank(x)))
    try:
        return lazy.asarray(np.asarray(np.linalg.matrix_rank(x, rtol=rtol)))
    except TypeError:
        warnings.warn(
            "matrix_rank fallback cannot apply rtol as a relative tolerance with "
            "this NumPy version; using it as an absolute tolerance.",
            RuntimeWarning,
            stacklevel=2,
        )
        return lazy.asarray(np.asarray(np.linalg.matrix_rank(x, tol=rtol)))


def pinv(x, /, *, rtol=None):
    x = _warn_compute(x, "pinv")
    x = to_numpy(lazy.asarray(x))
    if rtol is None:
        return lazy.asarray(np.linalg.pinv(x))
    try:
        return lazy.asarray(np.linalg.pinv(x, rtol=rtol))
    except TypeError:
        return lazy.asarray(np.linalg.pinv(x, rcond=rtol))


def qr(x, /, *, mode="reduced"):
    x = _warn_compute(x, "qr")
    x = to_numpy(lazy.asarray(x))
    return lazy.asarray(np.linalg.qr(x, mode=mode))


def slogdet(x, /):
    x = _warn_compute(x, "slogdet")
    x = to_numpy(lazy.asarray(x))
    return lazy.asarray(np.linalg.slogdet(x))


def solve(x1, x2, /):
    x1 = _warn_compute(x1, "solve")
    x2 = _warn_compute(x2, "solve")
    try:
        x2_np = to_numpy(lazy.asarray(x2))
        return lazy.asarray(
            scipy_sparse_linalg.spsolve(to_scipy(lazy.asarray(x1)), x2_np)
        )
    except Exception:
        pass
    x1 = to_numpy(lazy.asarray(x1))
    x2 = to_numpy(lazy.asarray(x2))
    return lazy.asarray(np.linalg.solve(x1, x2))


def svd(x, /, *, full_matrices=True, k=None, rtol=None, atol=None):
    x = _warn_compute(x, "svd")
    if k is not None:
        try:
            kwargs = {
                "k": k,
                "return_singular_vectors": True,
            }
            if rtol is not None and atol is not None:
                warnings.warn(
                    "svd sparse fallback supports only one tolerance; "
                    "using min(rtol, atol).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                tol = builtins.min(rtol, atol)
            elif rtol is not None:
                tol = rtol
            else:
                tol = atol
            if tol is not None:
                kwargs["tol"] = tol
            return lazy.asarray(
                scipy_sparse_linalg.svds(to_scipy(lazy.asarray(x)), **kwargs)
            )
        except Exception:
            pass
    if rtol is not None or atol is not None:
        warnings.warn(
            "svd dense fallback does not support rtol or atol; "
            "ignoring tolerance arguments.",
            RuntimeWarning,
            stacklevel=2,
        )
    x = to_numpy(lazy.asarray(x))
    return lazy.asarray(np.linalg.svd(x, full_matrices=full_matrices))


def svdvals(x, /, *, k=None, rtol=None, atol=None):
    x = _warn_compute(x, "svdvals")
    if k is not None:
        try:
            kwargs = {
                "k": k,
                "return_singular_vectors": False,
            }
            if rtol is not None and atol is not None:
                warnings.warn(
                    "svdvals sparse fallback supports only one tolerance; "
                    "using min(rtol, atol).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                tol = builtins.min(rtol, atol)
            elif rtol is not None:
                tol = rtol
            else:
                tol = atol
            if tol is not None:
                kwargs["tol"] = tol
            return lazy.asarray(
                scipy_sparse_linalg.svds(to_scipy(lazy.asarray(x)), **kwargs)
            )
        except Exception:
            pass
    if rtol is not None or atol is not None:
        warnings.warn(
            "svdvals dense fallback does not support rtol or atol; "
            "ignoring tolerance arguments.",
            RuntimeWarning,
            stacklevel=2,
        )
    x = to_numpy(lazy.asarray(x))
    svdvals_func = getattr(np.linalg, "svdvals", None)
    if svdvals_func is None:
        return lazy.asarray(np.linalg.svd(x, compute_uv=False))
    return lazy.asarray(svdvals_func(x))


def fft(x, /, *, n=None, axis=-1, norm=None):
    x = _warn_compute(x, "fft")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.fft(x, n=n, axis=axis, norm=norm))
    except Exception:
        return lazy.asarray(np.fft.fft(x, n=n, axis=axis, norm=norm))


def ifft(x, /, *, n=None, axis=-1, norm=None):
    x = _warn_compute(x, "ifft")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.ifft(x, n=n, axis=axis, norm=norm))
    except Exception:
        return lazy.asarray(np.fft.ifft(x, n=n, axis=axis, norm=norm))


def fftn(x, /, *, s=None, axes=None, norm=None):
    x = _warn_compute(x, "fftn")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.fftn(x, s=s, axes=axes, norm=norm))
    except Exception:
        return lazy.asarray(np.fft.fftn(x, s=s, axes=axes, norm=norm))


def ifftn(x, /, *, s=None, axes=None, norm=None):
    x = _warn_compute(x, "ifftn")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.ifftn(x, s=s, axes=axes, norm=norm))
    except Exception:
        return lazy.asarray(np.fft.ifftn(x, s=s, axes=axes, norm=norm))


def rfft(x, /, *, n=None, axis=-1, norm=None):
    x = _warn_compute(x, "rfft")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.rfft(x, n=n, axis=axis, norm=norm))
    except Exception:
        return lazy.asarray(np.fft.rfft(x, n=n, axis=axis, norm=norm))


def irfft(x, /, *, n=None, axis=-1, norm=None):
    x = _warn_compute(x, "irfft")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.irfft(x, n=n, axis=axis, norm=norm))
    except Exception:
        return lazy.asarray(np.fft.irfft(x, n=n, axis=axis, norm=norm))


def rfftn(x, /, *, s=None, axes=None, norm=None):
    x = _warn_compute(x, "rfftn")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.rfftn(x, s=s, axes=axes, norm=norm))
    except Exception:
        return lazy.asarray(np.fft.rfftn(x, s=s, axes=axes, norm=norm))


def irfftn(x, /, *, s=None, axes=None, norm=None):
    x = _warn_compute(x, "irfftn")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.irfftn(x, s=s, axes=axes, norm=norm))
    except Exception:
        return lazy.asarray(np.fft.irfftn(x, s=s, axes=axes, norm=norm))


def hfft(x, /, *, n=None, axis=-1, norm=None):
    x = _warn_compute(x, "hfft")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.hfft(x, n=n, axis=axis, norm=norm))
    except Exception:
        return lazy.asarray(np.fft.hfft(x, n=n, axis=axis, norm=norm))


def ihfft(x, /, *, n=None, axis=-1, norm=None):
    x = _warn_compute(x, "ihfft")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.ihfft(x, n=n, axis=axis, norm=norm))
    except Exception:
        return lazy.asarray(np.fft.ihfft(x, n=n, axis=axis, norm=norm))


def fftshift(x, /, *, axes=None):
    x = _warn_compute(x, "fftshift")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.fftshift(x, axes=axes))
    except Exception:
        return lazy.asarray(np.fft.fftshift(x, axes=axes))


def ifftshift(x, /, *, axes=None):
    x = _warn_compute(x, "ifftshift")
    x = to_numpy(lazy.asarray(x))
    try:
        return lazy.asarray(scipy_fft.ifftshift(x, axes=axes))
    except Exception:
        return lazy.asarray(np.fft.ifftshift(x, axes=axes))


def fftfreq(n, /, *, d=1.0, device=None):
    try:
        result = scipy_fft.fftfreq(n, d=d)
    except Exception:
        result = np.fft.fftfreq(n, d=d)
    return lazy.asarray(result, device=device)


def rfftfreq(n, /, *, d=1.0, device=None):
    try:
        result = scipy_fft.rfftfreq(n, d=d)
    except Exception:
        result = np.fft.rfftfreq(n, d=d)
    return lazy.asarray(result, device=device)


def matrix_power(x, n, /):
    """
    Computes the power of a matrix.
    """
    if isinstance(x, lazy.LazyTensor) and not (isinstance(n, int) and n < 0):
        return lazy.matrix_power(x, n)
    if isinstance(n, int) and n < 0:
        return matrix_power(inv(x), -n)
    try:
        return lazy.asarray(
            to_numpy(scipy_sparse_linalg.matrix_power(to_scipy(lazy.asarray(x)), n))
        )
    except Exception:
        pass
    return compute(lazy.matrix_power(x, n))


def matrix_norm(x, /, *, keepdims=False, ord="fro"):
    if isinstance(x, lazy.LazyTensor):
        try:
            return lazy.matrix_norm(x, keepdims=keepdims, ord=ord)
        except NotImplementedError:
            pass
    x = _warn_compute(x, "matrix_norm")
    try:
        result = scipy_sparse_linalg.norm(
            to_scipy(lazy.asarray(x)),
            ord=ord,
            axis=(-2, -1),
        )
        if keepdims:
            result = np.reshape(result, (1, 1))
        return lazy.asarray(np.asarray(result))
    except Exception:
        pass
    x = to_numpy(lazy.asarray(x))
    matrix_norm_fn = getattr(np.linalg, "matrix_norm", None)
    if matrix_norm_fn is None:
        result = np.linalg.norm(x, ord=ord, axis=(-2, -1), keepdims=keepdims)
    else:
        result = matrix_norm_fn(x, keepdims=keepdims, ord=ord)
    return lazy.asarray(np.asarray(result))


def bitwise_invert(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.bitwise_invert(x)
    return compute(lazy.bitwise_invert(x))


def bitwise_and(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.bitwise_and(x1, x2)
    return compute(lazy.bitwise_and(x1, x2))


def bitwise_left_shift(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.bitwise_left_shift(x1, x2)
    return compute(lazy.bitwise_left_shift(x1, x2))


def bitwise_or(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.bitwise_or(x1, x2)
    return compute(lazy.bitwise_or(x1, x2))


def bitwise_right_shift(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.bitwise_right_shift(x1, x2)
    return compute(lazy.bitwise_right_shift(x1, x2))


def bitwise_xor(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.bitwise_xor(x1, x2)
    return compute(lazy.bitwise_xor(x1, x2))


def truediv(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.truediv(x1, x2)
    return compute(lazy.truediv(x1, x2))


def floor_divide(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.floor_divide(x1, x2)
    return compute(lazy.floor_divide(x1, x2))


def mod(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.mod(x1, x2)
    return compute(lazy.mod(x1, x2))


def pow(x1, x2):
    return power(x1, x2)


def power(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.power(x1, x2)
    return compute(lazy.power(x1, x2))


def remainder(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.remainder(x1, x2)
    return compute(lazy.remainder(x1, x2))


def tensordot(x1, x2, /, *, axes: int | tuple[Sequence[int], Sequence[int]] = 2):
    """
    Computes the tensordot operation.

    Returns a LazyTensor if either x1 or x2 is a LazyTensor.
    Otherwise, computes the result eagerly.
    """
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.tensordot(x1, x2, axes=axes)
    return compute(lazy.tensordot(x1, x2, axes=axes))


def vecdot(x1, x2, /, *, axis=-1):
    """
    Computes the (vector) dot product of two arrays.

    Parameters
    ----------
    x1: array
        The first input tensor.
    x2: array
        The second input tensor.
    axis: int, optional
        The axis along which to compute the dot product. Default is -1 (last axis).

    Returns
    -------
    out: array
        A tensor containing the dot product of `x1` and `x2` along the specified axis.
    """
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.vecdot(x1, x2, axis=axis)
    return compute(lazy.vecdot(x1, x2, axis=axis))


def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
    if isinstance(x, lazy.LazyTensor):
        return lazy.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    return compute(lazy.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord))


def any(x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.any(x, axis=axis, keepdims=keepdims)
    return compute(lazy.any(x, axis=axis, keepdims=keepdims))


def all(x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.all(x, axis=axis, keepdims=keepdims)
    return compute(lazy.all(x, axis=axis, keepdims=keepdims))


def real(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.real(x)
    return compute(lazy.real(x))


def imag(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.imag(x)
    return compute(lazy.imag(x))


def conj(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.conj(x)
    return compute(lazy.conj(x))


def min(x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.min(x, axis=axis, keepdims=keepdims)
    return compute(lazy.min(x, axis=axis, keepdims=keepdims))


def minimum(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.minimum(x1, x2)
    return compute(lazy.minimum(x1, x2))


def max(x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.max(x, axis=axis, keepdims=keepdims)
    return compute(lazy.max(x, axis=axis, keepdims=keepdims))


def maximum(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.maximum(x1, x2)
    return compute(lazy.maximum(x1, x2))


def clip(x, /, min=None, max=None):
    if (
        isinstance(x, lazy.LazyTensor)
        or isinstance(min, lazy.LazyTensor)
        or isinstance(max, lazy.LazyTensor)
    ):
        return lazy.clip(x, min=min, max=max)
    return compute(lazy.clip(x, min=min, max=max))


def sqrt(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.sqrt(x)
    return compute(lazy.sqrt(x))


def square(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.square(x)
    return compute(lazy.square(x))


def signbit(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.signbit(x)
    return compute(lazy.signbit(x))


def sign(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.sign(x)
    return compute(lazy.sign(x))


# manipulation functions:
# https://data-apis.org/array-api/2024.12/API_specification/manipulation_functions.html


def _flatten_for_concat(array):
    array = lazy.asarray(array)
    size = functools.reduce(operator.mul, array.shape, 1)
    if hasattr(array, "reshape"):
        return array.reshape((size,))
    if hasattr(array, "to_numpy"):
        return lazy.asarray(np.asarray(array.to_numpy()).reshape((size,)))
    return lazy.asarray(np.asarray(array).reshape((size,)))


def concat(arrays, /, *, axis: int | None = 0):
    arrays = tuple(arrays)
    if builtins.any(isinstance(array, lazy.LazyTensor) for array in arrays):
        return lazy.concat(arrays, axis=axis)
    if axis is None:
        arrays = tuple(_flatten_for_concat(array) for array in arrays)
        axis = 0
    return compute(lazy.concat(arrays, axis=axis))


def broadcast_to(x, /, shape: Sequence[int]):
    """
    Broadcasts an array to a new shape.

    Parameters
    ----------
    x: array
        The input tensor to be broadcasted.
    shape: Sequence[int]
        The target shape to which the input tensor should be broadcasted.

    Returns
    -------
    out: array
        A tensor with the same data as `x`, but with the specified shape.
    """
    shape = tuple(shape)  # Ensure shape is a tuple for consistency
    if isinstance(x, lazy.LazyTensor):
        return lazy.broadcast_to(x, shape=shape)
    return compute(lazy.broadcast_to(x, shape=shape))


def broadcast_arrays(*args):
    """
    Broadcasts one or more arrays against one another.

    Parameters
    ----------
    *args: array
        an arbitrary number of to-be broadcasted arrays.

    Returns
    -------
    out: List[array]
        a list of broadcasted arrays. Each array has the same shape.
        Element types are preserved.
    """
    if builtins.any(isinstance(arg, lazy.LazyTensor) for arg in args):
        return lazy.broadcast_arrays(*args)
    # compute can take in a list of LazyTensors
    return compute(lazy.broadcast_arrays(*args))


def meshgrid(*arrays, indexing: str = "xy"):
    if builtins.any(isinstance(arr, lazy.LazyTensor) for arr in arrays):
        return lazy.meshgrid(*arrays, indexing=indexing)
    return compute(lazy.meshgrid(*arrays, indexing=indexing))


def moveaxis(x, source: int | tuple[int, ...], destination: int | tuple[int, ...], /):
    """
    Moves array axes (dimensions) to new positions,
    while leaving other axes in their original positions.

    Args
    ---------
    - x (array) - input array.
    - source - Axes to move.
    - destination - indices defining the desired
    positions for each respective source axis index.

    Returns
    --------
    - out (array) - an array containing reordered axes.
    """
    if isinstance(x, lazy.LazyTensor):
        return lazy.moveaxis(x, source, destination)
    return compute(lazy.moveaxis(x, source, destination))


# trigonometric functions:
def sin(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.sin(x)
    return compute(lazy.sin(x))


def sinh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.sinh(x)
    return compute(lazy.sinh(x))


def cos(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.cos(x)
    return compute(lazy.cos(x))


def cosh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.cosh(x)
    return compute(lazy.cosh(x))


def tan(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.tan(x)
    return compute(lazy.tan(x))


def tanh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.tanh(x)
    return compute(lazy.tanh(x))


def asin(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.asin(x)
    return compute(lazy.asin(x))


def asinh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.asinh(x)
    return compute(lazy.asinh(x))


def acos(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.acos(x)
    return compute(lazy.acos(x))


def acosh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.acosh(x)
    return compute(lazy.acosh(x))


def atan(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.atan(x)
    return compute(lazy.atan(x))


def hypot(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.hypot(x1, x2)
    return compute(lazy.hypot(x1, x2))


def atanh(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.atanh(x)
    return compute(lazy.atanh(x))


def atan2(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.atan2(x1, x2)
    return compute(lazy.atan2(x1, x2))


def exp(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.exp(x)
    return compute(lazy.exp(x))


def expm1(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.expm1(x)
    return compute(lazy.expm1(x))


def log(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.log(x)
    return compute(lazy.log(x))


def log1p(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.log1p(x)
    return compute(lazy.log1p(x))


def log2(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.log2(x)
    return compute(lazy.log2(x))


def log10(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.log10(x)
    return compute(lazy.log10(x))


def logaddexp(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.logaddexp(x1, x2)
    return compute(lazy.logaddexp(x1, x2))


def copysign(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.copysign(x1, x2)
    return compute(lazy.copysign(x1, x2))


def count_nonzero(
    x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.count_nonzero(x, axis=axis, keepdims=keepdims)
    return compute(lazy.count_nonzero(x, axis=axis, keepdims=keepdims))


def count_nonfill(
    x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.count_nonfill(x, axis=axis, keepdims=keepdims)
    return compute(lazy.count_nonfill(x, axis=axis, keepdims=keepdims))


def nextafter(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.nextafter(x1, x2)
    return compute(lazy.nextafter(x1, x2))


def isfinite(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.isfinite(x)
    return compute(lazy.isfinite(x))


def isinf(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.isinf(x)
    return compute(lazy.isinf(x))


def isnan(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.isnan(x)
    return compute(lazy.isnan(x))


def iscomplexobj(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.iscomplexobj(x)
    return compute(lazy.iscomplexobj(x))


def logical_and(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.logical_and(x1, x2)
    return compute(lazy.logical_and(x1, x2))


def logical_or(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.logical_or(x1, x2)
    return compute(lazy.logical_or(x1, x2))


def logical_xor(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.logical_xor(x1, x2)
    return compute(lazy.logical_xor(x1, x2))


def logical_not(x):
    if isinstance(x, lazy.LazyTensor):
        return lazy.logical_not(x)
    return compute(lazy.logical_not(x))


def less(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.less(x1, x2)
    return compute(lazy.less(x1, x2))


def less_equal(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.less_equal(x1, x2)
    return compute(lazy.less_equal(x1, x2))


def greater(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.greater(x1, x2)
    return compute(lazy.greater(x1, x2))


def greater_equal(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.greater_equal(x1, x2)
    return compute(lazy.greater_equal(x1, x2))


def equal(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.equal(x1, x2)
    return compute(lazy.equal(x1, x2))


def same(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.same(x1, x2)
    return compute(lazy.same(x1, x2))


def not_equal(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.not_equal(x1, x2)
    return compute(lazy.not_equal(x1, x2))


def not_same(x1, x2):
    if isinstance(x1, lazy.LazyTensor) or isinstance(x2, lazy.LazyTensor):
        return lazy.not_same(x1, x2)
    return compute(lazy.not_same(x1, x2))


def where(condition, x1, x2):
    if (
        isinstance(condition, lazy.LazyTensor)
        or isinstance(x1, lazy.LazyTensor)
        or isinstance(x2, lazy.LazyTensor)
    ):
        return lazy.where(condition, x1, x2)
    return compute(lazy.where(condition, x1, x2))


def mean(x, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    if isinstance(x, lazy.LazyTensor):
        return lazy.mean(x, axis=axis, keepdims=keepdims)
    return compute(lazy.mean(x, axis=axis, keepdims=keepdims))


def reshape(x, /, shape: tuple, *, copy=None):
    if isinstance(x, lazy.LazyTensor):
        return lazy.reshape(x, shape, copy=copy)
    if not hasattr(x, "reshape"):
        raise NotImplementedError(f"Object of type {type(x)} does not support reshape")
    return x.reshape(shape, copy=copy)


def var(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.var(x, axis=axis, correction=correction, keepdims=keepdims)
    return compute(lazy.var(x, axis=axis, correction=correction, keepdims=keepdims))


def std(
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: bool = False,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.std(x, axis=axis, correction=correction, keepdims=keepdims)
    return compute(lazy.std(x, axis=axis, correction=correction, keepdims=keepdims))


def einop(prgm: str, /, **kwargs):
    """Execute an einsum expression using the specified array framework.

    This function parses and executes einsum-like expressions with extended syntax
    that supports various operations beyond traditional Einstein summation notation.

    Args:
        prgm (str): Einsum program string specifying the computation. The syntax
            supports:
            - Assignment: "C[i,j] = A[i,j] + B[j,i]"
            - Increment: "C[i,j] += A[i,k] * B[k,j]"
            - Reductions: "C[i] += A[i,j]", "C[i] max= A[i,j]", "C[i] &= A[i,j]"
            - Arithmetic operations: +, -, *, /, //, %, **
            - Comparison operations: ==, !=, <, <=, >, >=
            - Logical operations: and, or, not
            - Bitwise operations: &, |, ^, <<, >>
            - Function calls and complex expressions with parentheses
            - Mathematical functions: abs, sqrt, exp, log, sin, cos, tan, etc.
            - Literal values: integers, floats, booleans, and complex numbers
            - Python operator precedence and parentheses for grouping
        **kwargs: Named arrays referenced in the einsum expression. The keys
            should match the tensor names used in the program string.

    Returns:
        The result array from executing the einsum expression.

    Examples:
        >>> import numpy as np
        >>> A = np.random.rand(3, 4)
        >>> B = np.random.rand(4, 3)
        >>> # Matrix addition with transpose
        >>> C = einop("C[i,j] = A[i,j] + B[j,i]", A=A, B=B)
        >>> # Matrix multiplication
        >>> D = einop("D[i,j] += A[i,k] * B[k,j]", A=A, B=B)
        >>> # Min-Plus multiplication with shift
        >>> E = einop("E[i] min= A[i,k] + D[k,j] << 1", A=A, D=D)
    """
    if builtins.any(isinstance(v, lazy.LazyTensor) for v in kwargs.values()):
        return lazy.einop(prgm, **kwargs)
    return compute(lazy.einop(prgm, **kwargs))


def einsum(*args, **kwargs):
    """
    einsum(subscripts, *operands)

    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional,
    linear algebraic array operations can be represented in a simple fashion.
    In *implicit* mode `einsum` computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical Einstein
    summation operations, by disabling, or forcing summation over specified
    subscript labels.

    See the notes and examples for clarification.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : list of array_like
        These are the arrays for the operation.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    Notes
    -----
    The Einstein summation convention can be used to compute
    many multi-dimensional, linear algebraic array operations. `einsum`
    provides a succinct way of representing these.

    A non-exhaustive list of these operations,
    which can be computed by `einsum`, is shown below along with examples:

    * Trace of an array, :py:func:`numpy.trace`.
    * Return a diagonal, :py:func:`numpy.diag`.
    * Array axis summations, :py:func:`numpy.sum`.
    * Transpositions and permutations, :py:func:`numpy.transpose`.
    * Matrix multiplication and dot product, :py:func:`numpy.matmul`
        :py:func:`numpy.dot`.
    * Vector inner and outer products, :py:func:`numpy.inner`
        :py:func:`numpy.outer`.
    * Broadcasting, element-wise and scalar multiplication,
        :py:func:`numpy.multiply`.
    * Tensor contractions, :py:func:`numpy.tensordot`.
    * Chained array operations, in efficient calculation order,
        :py:func:`numpy.einsum_path`.

    The subscripts string is a comma-separated list of subscript labels,
    where each label refers to a dimension of the corresponding operand.
    Whenever a label is repeated it is summed, so ``np.einsum('i,i', a, b)``
    is equivalent to :py:func:`np.inner(a,b) <numpy.inner>`. If a label
    appears only once, it is not summed, so ``np.einsum('i', a)``
    produces a view of ``a`` with no changes. A further example
    ``np.einsum('ij,jk', a, b)`` describes traditional matrix multiplication
    and is equivalent to :py:func:`np.matmul(a,b) <numpy.matmul>`.
    Repeated subscript labels in one operand take the diagonal.
    For example, ``np.einsum('ii', a)`` is equivalent to
    :py:func:`np.trace(a) <numpy.trace>`.

    In *implicit mode*, the chosen subscripts are important
    since the axes of the output are reordered alphabetically.  This
    means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
    ``np.einsum('ji', a)`` takes its transpose. Additionally,
    ``np.einsum('ij,jk', a, b)`` returns a matrix multiplication, while,
    ``np.einsum('ij,jh', a, b)`` returns the transpose of the
    multiplication since subscript 'h' precedes subscript 'i'.

    In *explicit mode* the output can be directly controlled by
    specifying output subscript labels.  This requires the
    identifier '->' as well as the list of output subscript labels.
    This feature increases the flexibility of the function since
    summing can be disabled or forced when required. The call
    ``np.einsum('i->', a)`` is like :py:func:`np.sum(a) <numpy.sum>`
    if ``a`` is a 1-D array, and ``np.einsum('ii->i', a)``
    is like :py:func:`np.diag(a) <numpy.diag>` if ``a`` is a square 2-D array.
    The difference is that `einsum` does not allow broadcasting by default.
    Additionally ``np.einsum('ij,jh->ih', a, b)`` directly specifies the
    order of the output subscript labels and therefore returns matrix
    multiplication, unlike the example above in implicit mode.

    To enable and control broadcasting, use an ellipsis.  Default
    NumPy-style broadcasting is done by adding an ellipsis
    to the left of each term, like ``np.einsum('...ii->...i', a)``.
    ``np.einsum('...i->...', a)`` is like
    :py:func:`np.sum(a, axis=-1) <numpy.sum>` for array ``a`` of any shape.
    To take the trace along the first and last axes,
    you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix
    product with the left-most indices instead of rightmost, one can do
    ``np.einsum('ij...,jk...->ik...', a, b)``.

    `einsum` also provides an alternative way to provide the subscripts and
    operands as ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``.
    If the output shape is not provided in this format `einsum` will be
    calculated in implicit mode, otherwise it will be performed explicitly.
    The examples below have corresponding `einsum` calls with the two
    parameter methods.

    Examples
    --------
    >>> a = np.arange(25).reshape(5, 5)
    >>> b = np.arange(5)
    >>> c = np.arange(6).reshape(2, 3)

    Trace of a matrix:

    >>> np.einsum("ii", a)
    60
    >>> np.einsum(a, [0, 0])
    60
    >>> np.trace(a)
    60

    Extract the diagonal (requires explicit form):

    >>> np.einsum("ii->i", a)
    array([ 0,  6, 12, 18, 24])
    >>> np.einsum(a, [0, 0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> np.diag(a)
    array([ 0,  6, 12, 18, 24])

    Sum over an axis (requires explicit form):

    >>> np.einsum("ij->i", a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [0, 1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> np.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])

    For higher dimensional arrays summing a single axis can be done
    with ellipsis:

    >>> np.einsum("...j->...", a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [Ellipsis, 1], [Ellipsis])
    array([ 10,  35,  60,  85, 110])

    Compute a matrix transpose, or reorder any number of axes:

    >>> np.einsum("ji", c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum("ij->ji", c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum(c, [1, 0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])

    Vector inner products:

    >>> np.einsum("i,i", b, b)
    30
    >>> np.einsum(b, [0], b, [0])
    30
    >>> np.inner(b, b)
    30

    Matrix vector multiplication:

    >>> np.einsum("ij,j", a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum(a, [0, 1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> np.dot(a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum("...j,j", a, b)
    array([ 30,  80, 130, 180, 230])

    Broadcasting and scalar multiplication:

    >>> np.einsum("..., ...", 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(",ij", 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(3, [Ellipsis], c, [Ellipsis])
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.multiply(3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])

    Vector outer product:

    >>> np.einsum("i,j", np.arange(2) + 1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.einsum(np.arange(2) + 1, [0], b, [1])
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.outer(np.arange(2) + 1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])

    Tensor contraction:

    >>> a = np.arange(60.0).reshape(3, 4, 5)
    >>> b = np.arange(24.0).reshape(4, 3, 2)
    >>> np.einsum("ijk,jil->kl", a, b)
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> np.einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3])
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> np.tensordot(a, b, axes=([1, 0], [0, 1]))
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])

    Example of ellipsis use:

    >>> a = np.arange(6).reshape((3, 2))
    >>> b = np.arange(12).reshape((4, 3))
    >>> np.einsum("ki,jk->ij", a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum("ki,...k->i...", a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum("k...,jk", a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    """

    if builtins.any(isinstance(v, lazy.LazyTensor) for v in args):
        return lazy.einsum(*args, **kwargs)
    return compute(lazy.einsum(*args, **kwargs))
