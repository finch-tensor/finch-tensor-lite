from finchlite.galley.PhysicalOptimizer.representation import ElementData, DenseData, ExtrudeData, HollowData, RepeatData, SparseData, Representation
from typing import Any, Callable, List

def fill_value(rep: Representation) -> Any:
    """
    Returns the fill value for the representation.
    """
    if isinstance(rep, ElementData):
        return rep.fill_value
    elif hasattr(rep, 'lvl'):
        return fill_value(rep.lvl)
    else:
        raise ValueError(f"Unsupported representation: {type(rep)}")

def eltype(rep: Representation) -> type:
    """
    Returns the element type for the representation.
    """
    if isinstance(rep, ElementData):
        return rep.element_type
    elif hasattr(rep, 'lvl'):
        return eltype(rep.lvl)
    else:
        raise ValueError(f"Unsupported representation: {type(rep)}")

def data_rep(tns) -> Representation:
    """
    Returns the data representation for the tensor.
    """
    # scalar
    if isinstance(tns, type) and issubclass(tns, (int, float, complex)):
        return ElementData(fill_value = 0, element_type = tns)
    # tensor
    result = ElementData(fill_value(tns), eltype(tns))
    # TODO: handle other types of tensors, the jl version assumes the tensor is dense, so i ported, but may need to handle other cases.
    for _ in range(tns.ndim if hasattr(tns, 'ndim') else 0):
        result = DenseData(result)
    return result

def expanddims_rep(tns: Representation, dims: List[int]) -> Representation:
    """
    Expands the dimensions of the representation.
    """
    # checks that were present in the julia version. added here for now.
    assert len(set(dims)) == len(dims), "Dimensions must be unique"
    assert all(d in range(1, tns.ndims() + len(dims)) for d in dims), "Dimensions must be in range 1 to ndims(tns) + len(dims)"
    return _expanddims_rep_def(tns, len(dims) + (tns.ndims() if hasattr(tns, 'ndims') else 0), dims)

def _expanddims_rep_def(tns: Representation, dim: int, dims: List[int]) -> Representation:
    """
    Expands the dimensions of the representation.
    """
    if isinstance(tns, ElementData):
        if dim in dims:
            return ExtrudeData(_expanddims_rep_def(tns, dim - 1, dims))
        return tns
    elif isinstance(tns, HollowData):
        return HollowData(_expanddims_rep_def(tns.lvl, dim, dims))
    elif isinstance(tns, ExtrudeData):
        if dim in dims:
            return ExtrudeData(_expanddims_rep_def(tns, dim - 1, dims))
        return ExtrudeData(_expanddims_rep_def(tns.lvl, dim - 1, dims))
    elif isinstance(tns, SparseData):
        if dim in dims:
            return ExtrudeData(_expanddims_rep_def(tns, dim - 1, dims))
        return SparseData(_expanddims_rep_def(tns.lvl, dim - 1, dims))
    elif isinstance(tns, RepeatData):
        if dim in dims:
            return ExtrudeData(_expanddims_rep_def(tns, dim - 1, dims))
        return RepeatData(_expanddims_rep_def(tns.lvl, dim - 1, dims))
    elif isinstance(tns, DenseData):
        if dim in dims:
            return ExtrudeData(_expanddims_rep_def(tns, dim - 1, dims))
        return DenseData(_expanddims_rep_def(tns.lvl, dim - 1, dims))
    else:
        raise ValueError(f"Unsupported representation: {type(tns)}")

def map_rep(f: Callable, *args: Representation) -> Representation:
    """
    Predict sparsity pattern after applying the function to the representations.
    """
    if not args:
        raise ValueError("At least one argument is required")
    
    max_dims = max(arg.ndims() if hasattr(arg, 'ndims') else 0 for arg in args)
    padded_args = [_paddims_rep(arg, max_dims) for arg in args]

    return _map_rep_def(f, padded_args)
    
def _map_rep_def(f: Callable, args: List[Representation]) -> Representation:
    """
    Maps the function over the representations.
    """
    if any(isinstance(arg, HollowData) for arg in args):
        return _map_rep_def_hollow(f, args)
    elif any(isinstance(arg, SparseData) for arg in args):
        return _map_rep_def_sparse(f, args)
    elif any(isinstance(arg, DenseData) for arg in args):
        return _map_rep_def_dense(f, args)
    elif any(isinstance(arg, RepeatData) for arg in args):
        return _map_rep_def_repeat(f, args)
    elif any(isinstance(arg, ExtrudeData) for arg in args):
        return _map_rep_def_extrude(f, args)
    else:
        return _map_rep_def_element(f, args)

def _paddims_rep(rep: Representation, dims: int) -> Representation:
    while rep.ndims() < dims:
        rep = ExtrudeData(rep)
    return rep

def _map_rep_def_hollow(f: Callable, args: List[Representation]) -> Representation:
    """
    Maps the function over the representations.
    """
    children = [arg.lvl if isinstance(arg, HollowData) else arg for arg in args]
    lvl = _map_rep_def(f, children)
    if all(isinstance(arg, HollowData) for arg in args):
        return HollowData(lvl)
    else:
        return lvl

def _map_rep_child(rep: Representation) -> Representation:
    if hasattr(rep, 'lvl'):
        return rep.lvl
    return rep


def _map_rep_def_sparse(f: Callable, args: List[Representation]) -> Representation:
    """
    Maps the function over the representations.
    """
    children = [_map_rep_child(arg) for arg in args]
    lvl = _map_rep_def(f, children)
    if all(isinstance(arg, SparseData) for arg in args):
        return SparseData(lvl)
    else:
        # TODO: Julia can check if op preserves sparsity, but we don't have that yet. Assume Dense. Add symbolic check.
        return DenseData(lvl)

def _map_rep_def_dense(f: Callable, args: List[Representation]) -> Representation:
    """
    Maps the function over the representations.
    """
    children = [_map_rep_child(arg) for arg in args]
    lvl = _map_rep_def(f, children)
    return DenseData(lvl)

def _map_rep_def_repeat(f: Callable, args: List[Representation]) -> Representation:
    """
    Maps the function over the representations.
    """
    children = [_map_rep_child(arg) for arg in args]
    lvl = _map_rep_def(f, children)
    if all(isinstance(arg, RepeatData) for arg in args):
        return RepeatData(lvl)
    else:
        # TODO: Julia can check if op preserves repeat, but we don't have that yet. Assume Dense. Add symbolic check.
        return DenseData(lvl)

def _map_rep_def_extrude(f: Callable, args: List[Representation]) -> Representation:
    """
    Maps the function over the representations.
    """
    children = [_map_rep_child(arg) for arg in args]
    lvl = _map_rep_def(f, children)
    return ExtrudeData(lvl)

def _map_rep_def_element(f: Callable, args: List[Representation]) -> Representation:
    """
    Apply f to the fill values of the arguments. Infer the element type.
    """
    fill_values = f(*[fill_value(arg) for arg in args])
    element_type = type(fill_values)
    return ElementData(fill_values, element_type)


def aggregate_rep(op: Callable, init: Any, rep: Representation, dims: List[int]) -> Representation:
    """
    Aggregate the representation over the dimensions.
    """
    drops = tuple(reversed([i + 1 in dims for i in range(rep.ndims())]))
    return _aggregate_rep_def(op, init, rep, *drops)

def _aggregate_rep_def(op: Callable, init: Any, rep: Representation, *drops: bool) -> Representation:
    """
    Aggregate the representation over the dimensions.
    """
    if isinstance(rep, ElementData):
        return ElementData(init, type(init))

    if isinstance(rep, HollowData):
        return HollowData(_aggregate_rep_def(op, init, rep.lvl, *drops))

    if not drops:
        return rep
    
    drop, *rest = drops
    if isinstance(rep, SparseData):
        if drop:
            return _aggregate_rep_def(op, init, rep.lvl, *rest)
        else:
            inner_dim = _aggregate_rep_def(op, init, rep.lvl, *rest)
            if op(init, fill_value(rep)) == init:
                return SparseData(inner_dim)
            return DenseData(inner_dim)
    elif isinstance(rep, DenseData):
        if drop:
            return _aggregate_rep_def(op, init, rep.lvl, *rest)
        else:
            return DenseData(_aggregate_rep_def(op, init, rep.lvl, *rest))
    elif isinstance(rep, RepeatData):
        if drop:
            return _aggregate_rep_def(op, init, rep.lvl, *rest)
        return RepeatData(_aggregate_rep_def(op, init, rep.lvl, *rest))
    elif isinstance(rep, ExtrudeData):
        if drop:
            return _aggregate_rep_def(op, init, rep.lvl, *rest)
        return ExtrudeData(_aggregate_rep_def(op, init, rep.lvl, *rest))
    
    return rep

def dropdims_rep(rep: Representation, dims: List[int]) -> Representation:
    """
    Drop the dimensions of the representation.
    """
    return aggregate_rep(lambda x, y: x, fill_value(rep), rep, dims)

def permutedims_rep(rep: Representation, perm: List[int]) -> Representation:
    """
    Permute the dimensions of the representation.
    """
    if not perm or len(perm) == 0:
        return rep
    if isinstance(rep, HollowData):
        return HollowData(permutedims_rep(rep.lvl, perm))
    
    # TODO: Implement permutedims, lines 381-389 in optimizer.jl.
    # Would need collapse_rep, permutedims_rep_select_def, permutedims_rep_aggregate_def, lot of code. added placeholder for now.
    return rep