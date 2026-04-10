import numpy as np

from finchlite.algebra.ftype import fisinstance
from finchlite.algebra.tensor import TensorFType
from finchlite.finch_assembly.stages import AssemblyKernel, AssemblyLibrary
from finchlite.finch_einsum.stages import (
    EinsumEvaluator,
    EinsumLoader,
    compute_shape_vars,
)

from ..algebra import ffunc
from . import nodes as ein

nary_ops = {
    ffunc.add: "add",
    ffunc.mul: "multiply",
    ffunc.sub: "subtract",
    ffunc.truediv: "divide",
    ffunc.floordiv: "floor_divide",
    ffunc.mod: "remainder",
    ffunc.pow: "power",
    ffunc.eq: "equal",
    ffunc.ne: "not_equal",
    ffunc.lt: "less",
    ffunc.le: "less_equal",
    ffunc.gt: "greater",
    ffunc.ge: "greater_equal",
    ffunc.and_: "bitwise_and",
    ffunc.or_: "bitwise_or",
    ffunc.xor: "bitwise_xor",
    ffunc.lshift: "bitwise_left_shift",
    ffunc.rshift: "bitwise_right_shift",
    ffunc.logical_and: "logical_and",
    ffunc.logical_or: "logical_or",
    ffunc.logical_not: "logical_not",
    ffunc.min: "minimum",
    ffunc.max: "maximum",
}
unary_ops = {
    ffunc.pos: "positive",
    ffunc.neg: "negative",
    ffunc.invert: "bitwise_invert",
    ffunc.abs: "absolute",
    ffunc.sqrt: "sqrt",
    ffunc.exp: "exp",
    ffunc.log: "log",
    ffunc.log1p: "log1p",
    ffunc.log10: "log10",
    ffunc.log2: "log2",
    ffunc.sin: "sin",
    ffunc.cos: "cos",
    ffunc.tan: "tan",
    ffunc.sinh: "sinh",
    ffunc.cosh: "cosh",
    ffunc.tanh: "tanh",
    ffunc.arcsin: "arcsin",
    ffunc.arccos: "arccos",
    ffunc.arctan: "arctan",
    ffunc.arcsinh: "arcsinh",
    ffunc.arccosh: "arccosh",
    ffunc.arctanh: "arctanh",
}

reduction_ops = {
    ffunc.add: "sum",
    ffunc.mul: "prod",
    ffunc.and_: "all",
    ffunc.or_: "any",
    ffunc.min: "min",
    ffunc.max: "max",
    ffunc.logical_and: "all",
    ffunc.logical_or: "any",
}


class EinsumInterpreter(EinsumEvaluator):
    def __init__(self, xp=np):
        self.xp = xp

    def __call__(self, node, bindings=None):
        if bindings is None:
            bindings = {}
        bindings = {k: self.xp.asarray(v) for k, v in bindings.items()}
        machine = EinsumMachine(xp=self.xp, bindings=bindings.copy())
        return machine(node)


class PointwiseEinsumMachine:
    def __init__(self, xp, bindings, loops):
        self.xp = xp
        self.bindings = bindings
        self.loops = loops

    def __call__(self, node):
        xp = self.xp
        match node:
            case ein.Literal(val):
                return self.xp.full([1 for _ in self.loops], val)
            case ein.Alias(name):
                if node not in self.bindings:
                    raise ValueError(f"Unbound variable: {name}")
                return self.bindings[node]
            case ein.Call(ein.Literal(func), args):
                if len(args) == 1:
                    func = getattr(xp, unary_ops[func])
                else:
                    func = getattr(xp, nary_ops[func])
                vals = [self(arg) for arg in args]
                return func(*vals)
            case ein.Access(tns, idxs):
                assert len(idxs) == len(set(idxs))
                assert self.loops is not None
                perm = [idxs.index(idx) for idx in self.loops if idx in idxs]
                tns = self(tns)
                tns = xp.permute_dims(tns, perm)
                return xp.expand_dims(
                    tns,
                    [i for i in range(len(self.loops)) if self.loops[i] not in idxs],
                )
            case _:
                raise ValueError(f"Unknown einsum type: {type(node)}")


class EinsumMachine:
    def __init__(self, xp, bindings):
        self.xp = xp
        self.bindings = bindings

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
                ctx = PointwiseEinsumMachine(self.xp, self.bindings, loops)
                arg = ctx(arg)
                axis = tuple(i for i in range(len(loops)) if loops[i] not in idxs)
                if op != ffunc.overwrite:
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
