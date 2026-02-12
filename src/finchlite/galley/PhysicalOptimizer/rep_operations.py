import operator
from collections.abc import Callable
from typing import Any

from finchlite.galley.PhysicalOptimizer.representation import (
    DenseData,
    ElementData,
    ExtrudeData,
    HollowData,
    RepeatData,
    SparseData,
)

Representation = (
    ElementData | DenseData | ExtrudeData | HollowData | RepeatData | SparseData
)


def _preserves_sparsity(f: Callable, fill: float = 0.0) -> bool:
    """Check if operation keeps zeros as zeros."""
    if f in {operator.mul, operator.truediv, operator.floordiv, operator.mod}:
        return True

    try:
        result1 = f(1.0, fill)
        result2 = f(fill, 1.0)
        if result1 == fill or result2 == fill:
            return True
    except Exception:
        # just in case func is complex
        pass
    return False


def fill_value(rep: Representation) -> Any:
    """
    Returns the fill value for the representation.
    """
    if isinstance(rep, ElementData):
        return rep.fill_value
    if hasattr(rep, "lvl"):
        return fill_value(rep.lvl)
    raise ValueError(f"Unsupported representation: {type(rep)}")


def eltype(rep: Representation) -> type:
    """
    Returns the element type for the representation.
    """
    if isinstance(rep, ElementData):
        return rep.element_type
    if hasattr(rep, "lvl"):
        return eltype(rep.lvl)
    raise ValueError(f"Unsupported representation: {type(rep)}")


def data_rep(tns) -> Representation:
    """
    Returns the data representation for the tensor.
    """

    if isinstance(tns, type) and issubclass(tns, (int, float, complex)):
        return ElementData(fill_value=0, element_type=tns)

    if hasattr(tns, "dtype") and hasattr(tns, "ndim"):
        fill_val = 0
        elem_type = tns.dtype.type if hasattr(tns.dtype, "type") else type(tns.flat[0])
        result = ElementData(fill_value=fill_val, element_type=elem_type)
        for _ in range(tns.ndim):
            result = DenseData(result)
        return result

    result = ElementData(fill_value(tns), eltype(tns))
    for _ in range(tns.ndim if hasattr(tns, "ndim") else 0):
        result = DenseData(result)
    return result


def expanddims_rep(tns: Representation, dims: list[int]) -> Representation:
    """
    Expands the dimensions of the representation.
    """
    # checks that were present in the julia version. added here for now.
    assert len(set(dims)) == len(dims), "Dimensions must be unique"

    total_dims = tns.ndims() + len(dims)
    for d in dims:
        assert 1 <= d <= total_dims, f"Dimension {d} out of range"

    return _expanddims_rep_def(tns, total_dims, dims)


def _expanddims_rep_def(
    tns: Representation, dim: int, dims: list[int]
) -> Representation:
    """
    Expands the dimensions of the representation.
    """
    if isinstance(tns, ElementData):
        if dim in dims:
            return ExtrudeData(_expanddims_rep_def(tns, dim - 1, dims))
        return tns

    if isinstance(tns, HollowData):
        return HollowData(_expanddims_rep_def(tns.lvl, dim, dims))

    if dim in dims:
        return ExtrudeData(_expanddims_rep_def(tns, dim - 1, dims))

    child = _expanddims_rep_def(tns.lvl, dim - 1, dims)

    if isinstance(tns, ExtrudeData):
        return ExtrudeData(child)
    if isinstance(tns, SparseData):
        return SparseData(child)
    if isinstance(tns, RepeatData):
        return RepeatData(child)
    if isinstance(tns, DenseData):
        return DenseData(child)
    raise ValueError(f"Unsupported representation: {type(tns)}")


def map_rep(f: Callable, *args: Representation) -> Representation:
    """
    Predict sparsity pattern after applying the function to the representations.
    """
    if not args:
        raise ValueError("Need at least one argument")

    max_dims = max(arg.ndims() for arg in args)
    padded_args = []
    for arg in args:
        while arg.ndims() < max_dims:
            arg = ExtrudeData(arg)
        padded_args.append(arg)

    return _map_rep_def(f, padded_args)


def _map_rep_def(f: Callable, args: list[Representation]) -> Representation:
    """
    Maps the function over the representations.
    """
    if any(isinstance(arg, HollowData) for arg in args):
        return _map_rep_def_hollow(f, args)
    if any(isinstance(arg, SparseData) for arg in args):
        return _map_rep_def_sparse(f, args)
    if any(isinstance(arg, DenseData) for arg in args):
        return _map_rep_def_dense(f, args)
    if any(isinstance(arg, RepeatData) for arg in args):
        return _map_rep_def_repeat(f, args)
    if any(isinstance(arg, ExtrudeData) for arg in args):
        return _map_rep_def_extrude(f, args)

    return _map_rep_def_element(f, args)


def _map_rep_def_hollow(f: Callable, args: list[Representation]) -> Representation:
    """Maps the function over the representations."""
    children = []
    for arg in args:
        if isinstance(arg, HollowData):
            children.append(arg.lvl)
        else:
            children.append(arg)

    lvl = _map_rep_def(f, children)

    if all(isinstance(arg, HollowData) for arg in args):
        return HollowData(lvl)
    return lvl


def _map_rep_child(rep: Representation) -> Representation:
    if hasattr(rep, "lvl"):
        return rep.lvl
    return rep


def _map_rep_def_sparse(f: Callable, args: list[Representation]) -> Representation:
    children = [_map_rep_child(arg) for arg in args]
    lvl = _map_rep_def(f, children)

    if all(isinstance(arg, SparseData) for arg in args):
        return SparseData(lvl)

    if _preserves_sparsity(f, fill_value(lvl)):
        return SparseData(lvl)

    return DenseData(lvl)


def _map_rep_def_dense(f: Callable, args: list[Representation]) -> Representation:
    """
    Maps the function over the representations.
    """
    children = [_map_rep_child(arg) for arg in args]
    lvl = _map_rep_def(f, children)
    return DenseData(lvl)


def _map_rep_def_repeat(f: Callable, args: list[Representation]) -> Representation:
    children = [_map_rep_child(arg) for arg in args]
    lvl = _map_rep_def(f, children)

    if all(isinstance(arg, RepeatData) for arg in args):
        return RepeatData(lvl)

    if _preserves_sparsity(f, fill_value(lvl)):
        return RepeatData(lvl)

    return DenseData(lvl)


def _map_rep_def_extrude(f: Callable, args: list[Representation]) -> Representation:
    """
    Maps the function over the representations.
    """
    children = [_map_rep_child(arg) for arg in args]
    lvl = _map_rep_def(f, children)
    return ExtrudeData(lvl)


def _map_rep_def_element(f: Callable, args: list[Representation]) -> Representation:
    """
    Apply f to the fill values of the representations.
    """
    fill_values = [fill_value(arg) for arg in args]
    result_value = f(*fill_values)
    result_type = type(result_value)
    return ElementData(result_value, result_type)


def aggregate_rep(
    op: Callable, init: Any, rep: Representation, dims: list[int]
) -> Representation:
    """Aggregate representation over specified dimensions."""
    drops = []
    for i in range(rep.ndims()):
        dim_num = i + 1
        should_drop = dim_num in dims
        drops.append(should_drop)

    drops = tuple(reversed(drops))
    return _aggregate_rep_def(op, init, rep, *drops)


def _aggregate_rep_def(
    op: Callable, init: Any, rep: Representation, *drops: bool
) -> Representation:
    """
    Aggregate the representation over the dimensions.
    """
    if isinstance(rep, ElementData):
        return ElementData(init, type(init))

    if isinstance(rep, HollowData):
        return HollowData(_aggregate_rep_def(op, init, rep.lvl, *drops))

    if not drops:
        return rep

    drop = drops[0]
    rest = drops[1:]

    if isinstance(rep, SparseData):
        if drop:
            return _aggregate_rep_def(op, init, rep.lvl, *rest)
        inner_dim = _aggregate_rep_def(op, init, rep.lvl, *rest)
        if op(init, fill_value(rep)) == init:
            return SparseData(inner_dim)
        return DenseData(inner_dim)

    if isinstance(rep, DenseData):
        if drop:
            return _aggregate_rep_def(op, init, rep.lvl, *rest)
        return DenseData(_aggregate_rep_def(op, init, rep.lvl, *rest))

    if isinstance(rep, RepeatData):
        if drop:
            return _aggregate_rep_def(op, init, rep.lvl, *rest)
        return RepeatData(_aggregate_rep_def(op, init, rep.lvl, *rest))

    if isinstance(rep, ExtrudeData):
        if drop:
            return _aggregate_rep_def(op, init, rep.lvl, *rest)
        return ExtrudeData(_aggregate_rep_def(op, init, rep.lvl, *rest))

    return rep


def dropdims_rep(rep: Representation, dims: list[int]) -> Representation:
    """Drop dimensions by aggregating them away."""

    def keep_last(x, y):
        return y

    return aggregate_rep(keep_last, fill_value(rep), rep, dims)


def collapse_rep(rep: Representation) -> Representation:
    if isinstance(rep, ElementData):
        return rep
    if isinstance(rep, HollowData):
        child = collapse_rep(rep.lvl)
        if isinstance(child, HollowData):
            return collapse_rep(child)
        return HollowData(child)
    if isinstance(rep, DenseData):
        child = collapse_rep(rep.lvl)
        if isinstance(child, HollowData):
            return collapse_rep(SparseData(child.lvl))
        return DenseData(child)
    if isinstance(rep, SparseData):
        child = collapse_rep(rep.lvl)
        if isinstance(child, HollowData):
            return collapse_rep(SparseData(child.lvl))
        return SparseData(child)
    if isinstance(rep, ExtrudeData):
        child = collapse_rep(rep.lvl)
        if isinstance(child, HollowData):
            return HollowData(collapse_rep(ExtrudeData(child.lvl)))
        return ExtrudeData(child)
    if isinstance(rep, RepeatData):
        child = collapse_rep(rep.lvl)
        if isinstance(child, HollowData):
            return collapse_rep(RepeatData(child.lvl))
        return RepeatData(child)
    return rep


def _permute_select(rep: Representation, drops: list[bool]) -> Representation:
    if isinstance(rep, ElementData):
        return rep
    if not drops:
        return rep
    drop = drops[0]
    rest = drops[1:]
    if isinstance(rep, SparseData):
        child = _permute_select(rep.lvl, rest)
        return HollowData(child) if drop else SparseData(child)
    if isinstance(rep, DenseData):
        child = _permute_select(rep.lvl, rest)
        return child if drop else DenseData(child)
    if isinstance(rep, ExtrudeData):
        child = _permute_select(rep.lvl, rest)
        return child if drop else ExtrudeData(child)
    if isinstance(rep, RepeatData):
        child = _permute_select(rep.lvl, rest)
        return child if drop else RepeatData(child)
    return rep


def _permute_aggregate(
    leaf: Representation, rep: Representation, drops: list[bool]
) -> Representation:
    if isinstance(rep, ElementData):
        return leaf
    if isinstance(rep, HollowData):
        return _permute_aggregate(leaf, rep.lvl, drops)
    if not drops:
        return leaf
    drop = drops[0]
    rest = drops[1:]
    if drop:
        return _permute_aggregate(leaf, rep.lvl, rest)
    child = _permute_aggregate(leaf, rep.lvl, rest)
    if isinstance(rep, SparseData):
        return SparseData(child)
    if isinstance(rep, DenseData):
        return DenseData(child)
    if isinstance(rep, ExtrudeData):
        return ExtrudeData(child)
    if isinstance(rep, RepeatData):
        return RepeatData(child)
    return child


def permutedims_rep(rep: Representation, perm: list[int]) -> Representation:
    if isinstance(rep, HollowData):
        return HollowData(permutedims_rep(rep.lvl, perm))
    if not perm or len(perm) <= 1:
        return rep
    if perm == list(range(len(perm))):
        return rep
    n = rep.ndims()
    last_dim = perm[-1]
    drops = [i != last_dim for i in range(n)]
    selected = collapse_rep(_permute_select(rep, list(reversed(drops))))
    new_perm = [p - (1 if p > last_dim else 0) for p in perm[:-1]]
    leaf = permutedims_rep(selected, new_perm)
    result = _permute_aggregate(leaf, rep, list(reversed(drops)))
    return collapse_rep(result)
