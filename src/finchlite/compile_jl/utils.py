# Helper functions for indexing support

import builtins

import numpy as np
from numpy._core.numeric import normalize_axis_index, normalize_axis_tuple

from .julia import jl


def expand_ellipsis(key: tuple, shape: tuple[int, ...]) -> tuple:
    ellipsis_pos = None
    key_without_ellipsis = []
    # first we need to find the ellipsis and confirm it's the only one
    for pos, idx in enumerate(key):
        if idx is Ellipsis:
            if ellipsis_pos is None:
                ellipsis_pos = pos
            else:
                raise IndexError("an index can only have a single ellipsis ('...')")
        else:
            key_without_ellipsis.append(idx)
    key = key_without_ellipsis

    # then we expand ellipsis with a full range
    if ellipsis_pos is not None:
        n_missing_idxs = len(shape) - builtins.sum(1 for k in key if k is not None)
        key = key[:ellipsis_pos] + [slice(None)] * n_missing_idxs + key[ellipsis_pos:]

    return tuple(key)


def add_missing_dims(key: tuple, shape: tuple[int, ...]) -> tuple:
    missing_dims = len(shape) - builtins.sum(1 for k in key if k is not None)
    return key + (slice(None),) * missing_dims


def _slice_plus_one(s: slice, size: int) -> range:
    step = s.step if s.step is not None else 1
    start_default = size if step < 0 else 1
    stop_default = 1 if step < 0 else size

    if s.start is not None:
        start = normalize_axis_index(s.start, size) + 1 if s.start < size else size
    else:
        start = start_default

    if s.stop is not None:
        stop_offset = 2 if step < 0 else 0
        stop = (
            normalize_axis_index(s.stop, size) + stop_offset if s.stop < size else size
        )
    else:
        stop = stop_default

    return jl.range(start=start, step=step, stop=stop)


def add_plus_one(key: tuple, shape: tuple[int, ...]) -> tuple:
    new_key = []
    sizes = iter(shape)
    for idx in key:
        if idx is None:
            new_key.append(jl.nothing)
            continue

        size = next(sizes)
        if isinstance(idx, int):
            new_key.append(normalize_axis_index(idx, size) + 1)
        elif isinstance(idx, slice):
            new_key.append(_slice_plus_one(idx, size))
        elif isinstance(idx, list | np.ndarray | tuple):
            idx = normalize_axis_tuple(idx, size)
            new_key.append(jl.Vector([i + 1 for i in idx]))
        else:
            new_key.append(idx)

    return tuple(new_key)
