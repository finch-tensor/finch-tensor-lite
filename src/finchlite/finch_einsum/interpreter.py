import operator
from typing import Any

import numpy as np

from finchlite.algebra.tensor import TensorFType
from finchlite.finch_assembly.stages import AssemblyKernel, AssemblyLibrary
from finchlite.finch_einsum.stages import (
    EinsumEvaluator,
    EinsumLoader,
    compute_shape_vars,
)
from finchlite.symbolic.ftype import fisinstance
from finchlite.symbolic.traversal import PostOrderDFS

from ..algebra import overwrite, promote_max, promote_min
from . import nodes as ein

nary_ops = {
    operator.add: "add",
    operator.mul: "multiply",
    operator.sub: "subtract",
    operator.truediv: "divide",
    operator.floordiv: "floor_divide",
    operator.mod: "remainder",
    operator.pow: "power",
    operator.eq: "equal",
    operator.ne: "not_equal",
    operator.lt: "less",
    operator.le: "less_equal",
    operator.gt: "greater",
    operator.ge: "greater_equal",
    operator.and_: "bitwise_and",
    operator.or_: "bitwise_or",
    operator.xor: "bitwise_xor",
    operator.lshift: "bitwise_left_shift",
    operator.rshift: "bitwise_right_shift",
    np.logical_and: "logical_and",
    np.logical_or: "logical_or",
    np.logical_not: "logical_not",
    promote_min: "minimum",
    promote_max: "maximum",
}
unary_ops = {
    operator.pos: "positive",
    operator.neg: "negative",
    operator.invert: "bitwise_invert",
    operator.abs: "absolute",
    np.sqrt: "sqrt",
    np.exp: "exp",
    np.log: "log",
    np.log1p: "log1p",
    np.log10: "log10",
    np.log2: "log2",
    np.sin: "sin",
    np.cos: "cos",
    np.tan: "tan",
    np.sinh: "sinh",
    np.cosh: "cosh",
    np.tanh: "tanh",
    np.arcsin: "arcsin",
    np.arccos: "arccos",
    np.arctan: "arctan",
    np.arcsinh: "arcsinh",
    np.arccosh: "arccosh",
    np.arctanh: "arctanh",
}

reduction_ops = {
    operator.add: "sum",
    operator.mul: "prod",
    operator.and_: "all",
    operator.or_: "any",
    promote_min: "min",
    promote_max: "max",
    np.logical_and: "all",
    np.logical_or: "any",
}


class EinsumInterpreter(EinsumEvaluator):
    def __init__(self, xp=np, verbose=False):
        self.xp = xp
        self.verbose = verbose

    def __call__(self, node, bindings=None):
        from ..tensor import SparseTensor

        if bindings is None:
            bindings = {}
        bindings = {
            k: v if isinstance(v, SparseTensor) else self.xp.asarray(v)
            for k, v in bindings.items()
        }
        machine = EinsumMachine(
            xp=self.xp, bindings=bindings.copy(), verbose=self.verbose
        )
        return machine(node)


class TensorEinsumMachine:
    def __init__(self, bindings):
        self.bindings = bindings

    def __call__(self, node):
        match node:
            case ein.Alias(name):
                if node not in self.bindings:
                    raise ValueError(f"Unbound variable: {name}")
                return self.bindings[node]
            case ein.Access(tns, (ein.Literal(dim),)):
                tns = self(tns)
                return tns[dim]
            case ein.GetAttr(obj, ein.Literal(attr)):
                obj = self(obj)
                if not hasattr(obj, attr):
                    raise ValueError(f"Object {obj} has no attribute {attr}")
                return getattr(obj, attr)


class PointwiseEinsumMachine:
    def __init__(self, xp, bindings, loops, dims, verbose):
        self.xp = xp
        self.bindings = bindings
        self.loops = loops
        self.dims = dims
        self.verbose = verbose
        self.tns_ctx = TensorEinsumMachine(bindings)

    def __call__(self, node):
        xp = self.xp
        match node:
            case ein.Literal(val):
                # If val is already an array, return it directly
                # (e.g., from recursive access).
                # Otherwise, broadcast the scalar to match the loop dimensions
                if hasattr(val, "ndim") and val.ndim > 0:
                    return val
                return self.xp.full([1 for _ in self.loops], val)
            case ein.Call(ein.Literal(func), args):
                if len(args) == 1:
                    func = getattr(xp, unary_ops[func])
                else:
                    func = getattr(xp, nary_ops[func])
                vals = [self(arg) for arg in args]
                # Promote to common dtype for Array API compatibility
                if len(vals) > 1:
                    common_dtype = xp.result_type(*vals)
                    vals = [xp.astype(v, common_dtype) for v in vals]
                return func(*vals)
            case ein.Index(_) as idx:
                tns = self.xp.arange(self.dims[idx])
                for _ in range(len(self.loops) - self.loops.index(idx) - 1):
                    tns = self.xp.expand_dims(tns, -1)
                return tns
            case ein.Access(tns, (ein.Literal(dim),)):
                tns = self.tns_ctx(tns)
                return tns[dim]
            case ein.Access(tns, idxs) if all(
                isinstance(idx, ein.Index) for idx in idxs
            ):
                assert self.loops is not None

                tns = self.tns_ctx(tns)
                assert len(idxs) == len(tns.shape)

                perm = [idxs.index(idx) for idx in self.loops if idx in idxs]
                if hasattr(tns, "ndim") and len(perm) < tns.ndim:
                    perm += list(range(len(perm), tns.ndim))

                tns = xp.permute_dims(tns, perm)  # permute the dimensions
                return xp.expand_dims(
                    tns,
                    [i for i in range(len(self.loops)) if self.loops[i] not in idxs],
                )
            case ein.Access(tns, idxs):
                assert self.loops is not None
                tns = self.tns_ctx(tns)
                evaled_items = tuple(self(idx) for idx in idxs)
                return tns[evaled_items]
            case ein.GetAttr(obj, ein.Literal(attr)):
                obj = self(obj)
                if not hasattr(obj, attr):
                    raise ValueError(f"Object {obj} has no attribute {attr}")
                return getattr(obj, attr)
            case _:
                raise ValueError(f"Unknown einsum type: {type(node)}")


class EinsumMachine:
    def __init__(self, xp, bindings, verbose):
        self.xp = xp
        self.bindings = bindings
        self.tns_ctx = TensorEinsumMachine(bindings)
        self.verbose = verbose

    def __call__(self, node):
        xp = self.xp
        match node:
            case ein.Plan(bodies):
                res = None
                for body in bodies:
                    res = self(body)
                return res
            case ein.Produces(args):
                for arg in args:
                    if arg not in self.bindings:
                        raise ValueError(f"Unbound variable: {arg}")
                return tuple(self.bindings[arg] for arg in args)
            case ein.Einsum(ein.Literal(op), tns, idxs, arg):
                loops = set(arg.get_idxs()).union(set(idxs))
                loops = sorted(loops, key=lambda x: x.name)
                dims: dict[ein.Index, Any] = {
                    idx: dim
                    for node in PostOrderDFS(arg)
                    if isinstance(node, ein.Access)
                    and not all(not isinstance(idx, ein.Index) for idx in node.idxs)
                    for idx, dim in zip(
                        node.idxs, self.tns_ctx(node.tns).shape, strict=True
                    )
                    if isinstance(idx, ein.Index)
                }
                ctx = PointwiseEinsumMachine(
                    self.xp, self.bindings, loops, dims, self.verbose
                )
                arg = ctx(arg)
                axis = tuple(i for i in range(len(loops)) if loops[i] not in idxs)
                if op != overwrite:
                    op = getattr(xp, reduction_ops[op])
                    val = op(arg, axis=axis)
                else:
                    val = arg
                    for i in sorted(axis, reverse=True):
                        val = xp.take(val, -1, axis=i)
                dropped = [idx for idx in loops if idx in idxs]
                axis = [dropped.index(idx) for idx in idxs]
                if tns in self.bindings:
                    if self.bindings[tns].ndim == 0:
                        self.bindings[tns][()] = val
                    else:
                        self.bindings[tns][:] = xp.permute_dims(val, axis)
                else:
                    self.bindings[tns] = xp.permute_dims(val, axis)
                return (self.bindings[tns],)
            case _:
                raise ValueError(f"Unknown einsum type: {type(node)}")


class MockEinsumKernel(AssemblyKernel):
    def __init__(self, prgm, bindings: dict[ein.Alias, TensorFType]):
        self.prgm = prgm
        self.bindings = bindings

    def __call__(self, *args):
        if len(args) != len(self.bindings):
            raise ValueError(
                f"Wrong number of arguments passed to kernel, "
                f"have {len(args)}, expected {len(self.bindings)}"
            )
        bindings = dict(zip(self.bindings.keys(), args, strict=True))
        for key in bindings:
            assert fisinstance(bindings[key], self.bindings[key])
        ctx = EinsumInterpreter()
        return ctx(self.prgm, bindings)


class MockEinsumLibrary(AssemblyLibrary):
    def __init__(self, prgm, bindings: dict[ein.Alias, TensorFType]):
        self.prgm = prgm
        self.bindings = bindings

    def __getattr__(self, name):
        if name == "main":
            return MockEinsumKernel(self.prgm, self.bindings)
        if name == "prgm":
            return self.prgm
        raise AttributeError(f"Unknown attribute {name} for InterpreterLibrary")


class MockEinsumLoader(EinsumLoader):
    def __init__(self):
        pass

    def __call__(
        self, prgm: ein.EinsumStatement, bindings: dict[ein.Alias, TensorFType]
    ) -> tuple[
        MockEinsumLibrary,
        dict[ein.Alias, TensorFType],
        dict[ein.Alias, tuple[ein.Index | None, ...]],
    ]:
        shape_vars = compute_shape_vars(prgm, bindings)
        return MockEinsumLibrary(prgm, bindings), bindings, shape_vars
