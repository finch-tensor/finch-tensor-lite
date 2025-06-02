from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from ..algebra import element_type, fill_value, fixpoint_type, return_type
from .nodes import (
    Aggregate,
    Alias,
    Deferred,
    Field,
    Immediate,
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    Subquery,
    Table,
)


@dataclass(eq=True, frozen=True)
class TableValue:
    tns: Any
    idxs: Iterable[Any]

    def __post_init__(self):
        if isinstance(self.tns, TableValue):
            raise ValueError("The tensor (tns) cannot be a TableValue")


class FinchLogicInterpreter:
    def __init__(self, *, make_tensor=np.full, verbose=False):
        self.verbose = verbose
        self.bindings = {}
        self.make_tensor = make_tensor  # Added make_tensor argument

    def __call__(self, node):
        # Example implementation for evaluating an expression
        if self.verbose:
            print(f"Evaluating: {node}")
        # Placeholder for actual logic
        match node:
            case Immediate(val):
                return val
            case Deferred(_):
                raise ValueError(
                    "The interpreter cannot evaluate a deferred node, a compiler might "
                    "generate code for it"
                )
            case Field(_):
                raise ValueError("Fields cannot be used in expressions")
            case Alias(_):
                alias = self.bindings.get(node, None)
                if alias is None:
                    raise ValueError(f"undefined tensor alias {node}")
                return alias
            case Table(Immediate(val), idxs):
                return TableValue(val, idxs)
            case MapJoin(Immediate(op), args):
                args = tuple(self(a) for a in args)
                dims = {}
                idxs = []
                for arg in args:
                    for idx, dim in zip(arg.idxs, arg.tns.shape, strict=True):
                        if idx in dims:
                            if dims[idx] != dim:
                                raise ValueError("Dimensions mismatched in map")
                        else:
                            idxs.append(idx)
                            dims[idx] = dim
                fill_val = op(*[fill_value(arg.tns) for arg in args])
                dtype = return_type(op, *[element_type(arg.tns) for arg in args])
                result = self.make_tensor(
                    tuple(dims[idx] for idx in idxs), fill_val, dtype=dtype
                )
                for crds in product(*[range(dims[idx]) for idx in idxs]):
                    idx_crds = dict(zip(idxs, crds, strict=True))
                    vals = [
                        arg.tns[*[idx_crds[idx] for idx in arg.idxs]] for arg in args
                    ]
                    result[*crds] = op(*vals)
                return TableValue(result, idxs)
            case Aggregate(Immediate(op), Immediate(init), arg, idxs):
                arg = self(arg)
                dtype = fixpoint_type(op, init, element_type(arg.tns))
                new_shape = tuple(
                    dim
                    for (dim, idx) in zip(arg.tns.shape, arg.idxs, strict=True)
                    if idx not in node.idxs
                )
                result = self.make_tensor(new_shape, init, dtype=dtype)
                for crds in product(*[range(dim) for dim in arg.tns.shape]):
                    out_crds = [
                        crd
                        for (crd, idx) in zip(crds, arg.idxs, strict=True)
                        if idx not in node.idxs
                    ]
                    result[*out_crds] = op(result[*out_crds], arg.tns[*crds])
                return TableValue(
                    result, [idx for idx in arg.idxs if idx not in node.idxs]
                )
            case Relabel(arg, idxs):
                arg = self(arg)
                if len(arg.idxs) != len(idxs):
                    raise ValueError("The number of indices in the relabel must match")
                return TableValue(arg.tns, idxs)
            case Reorder(arg, idxs):
                arg = self(arg)
                for idx, dim in zip(arg.idxs, arg.tns.shape, strict=True):
                    if idx not in idxs and dim != 1:
                        raise ValueError("Trying to drop a dimension that is not 1")
                arg_dims = dict(zip(arg.idxs, arg.tns.shape, strict=True))
                dims = [arg_dims.get(idx, 1) for idx in idxs]
                result = self.make_tensor(
                    dims, fill_value(arg.tns), dtype=arg.tns.dtype
                )
                for crds in product(*[range(dim) for dim in dims]):
                    node_crds = dict(zip(idxs, crds, strict=True))
                    in_crds = [node_crds.get(idx, 0) for idx in arg.idxs]
                    result[*crds] = arg.tns[*in_crds]
                return TableValue(result, idxs)
            case Query(lhs, rhs):
                rhs = self(rhs)
                self.bindings[lhs] = rhs
                return (rhs,)
            case Plan(bodies):
                res = ()
                for body in bodies:
                    res = self(body)
                return res
            case Produces(args):
                return tuple(self(arg).tns for arg in args)
            case Subquery(lhs, arg):
                res = self.bindings.get(lhs)
                if res is None:
                    res = self(arg)
                    self.bindings[lhs] = res
                return res
            case _:
                raise ValueError(f"Unknown expression type: {type(node)}")
