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
    def __init__(self, *, make_tensor=np.full):
        self.verbose = False
        self.bindings = {}
        self.make_tensor = make_tensor  # Added make_tensor argument

    def __call__(self, node):
        # Example implementation for evaluating an expression
        if self.verbose:
            print(f"Evaluating: {node}")
        # Placeholder for actual logic
        head = node.head()
        if head == Immediate:
            return node.val
        if head == Deferred:
            raise ValueError(
                "The interpreter cannot evaluate a deferred node, a compiler might "
                "generate code for it"
            )
        if head == Field:
            raise ValueError("Fields cannot be used in expressions")
        if head == Alias:
            if node in self.bindings:
                return self.bindings[node]
            raise ValueError(f"undefined tensor alias {node.val}")
        if head == Table:
            if node.tns.head() != Immediate:
                raise ValueError("The table data must be Immediate")
            return TableValue(node.tns.val, node.idxs)
        if head == MapJoin:
            if node.op.head() != Immediate:
                raise ValueError("The mapjoin operator must be Immediate")
            op = node.op.val
            args = list(map(self, node.args))
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
                vals = [arg.tns[*[idx_crds[idx] for idx in arg.idxs]] for arg in args]
                result[*crds] = op(*vals)
            return TableValue(result, idxs)
        if head == Aggregate:
            if node.op.head() != Immediate:
                raise ValueError("The aggregate operator must be Immediate")
            if node.init.head() != Immediate:
                raise ValueError("The aggregate initial value must be Immediate")
            arg = self(node.arg)
            init = node.init.val
            op = node.op.val
            dtype = fixpoint_type(op, init, element_type(arg.tns))
            new_shape = [
                dim
                for (dim, idx) in zip(arg.tns.shape, arg.idxs, strict=True)
                if idx not in node.idxs
            ]
            result = self.make_tensor(new_shape, init, dtype=dtype)
            for crds in product(*[range(dim) for dim in arg.tns.shape]):
                out_crds = [
                    crd
                    for (crd, idx) in zip(crds, arg.idxs, strict=True)
                    if idx not in node.idxs
                ]
                result[*out_crds] = op(result[*out_crds], arg.tns[*crds])
            return TableValue(result, [idx for idx in arg.idxs if idx not in node.idxs])
        if head == Relabel:
            arg = self(node.arg)
            if len(arg.idxs) != len(node.idxs):
                raise ValueError("The number of indices in the relabel must match")
            return TableValue(arg.tns, node.idxs)
        if head == Reorder:
            arg = self(node.arg)
            for idx, dim in zip(arg.idxs, arg.tns.shape, strict=True):
                if idx not in node.idxs and dim != 1:
                    raise ValueError("Trying to drop a dimension that is not 1")
            arg_dims = dict(zip(arg.idxs, arg.tns.shape, strict=True))
            dims = [arg_dims.get(idx, 1) for idx in node.idxs]
            result = self.make_tensor(dims, fill_value(arg.tns), dtype=arg.tns.dtype)
            for crds in product(*[range(dim) for dim in dims]):
                node_crds = dict(zip(node.idxs, crds, strict=True))
                in_crds = [node_crds.get(idx, 0) for idx in arg.idxs]
                result[*crds] = arg.tns[*in_crds]
            return TableValue(result, node.idxs)
        if head == Query:
            rhs = self(node.rhs)
            self.bindings[node.lhs] = rhs
            return (rhs,)
        if head == Plan:
            res = ()
            for body in node.bodies:
                res = self(body)
            return res
        if head == Produces:
            return tuple(self(arg).tns for arg in node.args)
        if head == Subquery:
            if node.lhs not in self.bindings:
                self.bindings[node.lhs] = self(node.arg)
            return self.bindings[node.lhs]
        raise ValueError(f"Unknown expression type: {head}")
