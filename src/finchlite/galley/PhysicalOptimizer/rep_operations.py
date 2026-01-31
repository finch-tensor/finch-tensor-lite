from finchlite.galley.PhysicalOptimizer.representation import ElementData, DenseData, Representation


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
    
    
