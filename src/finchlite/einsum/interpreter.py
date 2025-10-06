import operator
import builtins

import numpy as np

from . import nodes as ein

nary_ops = {
    operator.add: "add",
    operator.mul: "mul",
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
    builtins.min: "minimum",
    builtins.max: "maximum",
}

reduction_ops = {
    operator.add: "sum",
    operator.mul: "prod",
    operator.and_: "all",
    operator.or_: "any",
    builtins.min: "min",
    builtins.max: "max",
    np.logical_and: "all",
    np.logical_or: "any",
}


class EinsumInterpreter:
    def __init__(self, xp=None, bindings=None):
        if bindings is None:
            bindings = {}
        if xp is None:
            xp = np
        self.bindings = bindings
        self.xp = xp

    def __call__(self, prgm):
        xp = self.xp
        match prgm:
            case ein.Plan(bodies):
                res = None
                for body in bodies:
                    res = self(body)
                return res
            case ein.Produces(args):
                return tuple(self(arg) for arg in args)
            case ein.Einsum(op, tns, idxs, arg):
                # This is the main entry point for einsum execution
                loops = arg.get_idxs()
                assert set(idxs).issubset(loops)
                loops = sorted(loops)
                arg = self.eval(arg, loops)
                axis = tuple(i for i in range(len(loops)) if loops[i] not in idxs)
                if op is not None:
                    op = getattr(xp, reduction_ops.get(op))
                    val = op(arg, axis=axis)
                else:
                    assert set(idxs) == set(loops)
                    val = arg
                dropped = [idx for idx in loops if idx in idxs]
                axis = [dropped.index(idx) for idx in idxs]
                self.bindings[tns] = xp.transpose(val, axis)
                return (tns,)

    def eval(self, ex, loops):
        xp = self.xp
        match ex:
            case ein.Literal(val):
                return val
            case ein.Alias(name):
                return self.bindings[name]
            case ein.Call(func, args):
                func = self(func)
                if len(args) == 1:
                    func = getattr(xp, unary_ops[func])
                else:
                    func = getattr(xp, nary_ops[func])
                vals = [self.eval(arg, loops) for arg in args]
                return func(*vals)
            case ein.Access(tns, idxs):
                assert len(idxs) == len(set(idxs))
                perm = [idxs.index(idx) for idx in loops if idx in idxs]
                tns = self.bindings[tns]
                tns = xp.transpose(tns, perm)
                return xp.expand_dims(
                    tns, [i for i in range(len(loops)) if loops[i] not in self.idxs]
                )
