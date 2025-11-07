import operator

import numpy as np

from ..algebra import overwrite, promote_max, promote_min, TensorFType
from . import nodes as ein
from ..symbolic import ftype, gensym


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


class EinsumInterpreter:
    def __init__(self, xp=None, bindings=None, loops=None):
        if bindings is None:
            bindings = {}
        if xp is None:
            xp = np
        self.bindings = bindings
        self.xp = xp
        self.loops = loops

    def __call__(self, node):
        from ..tensor import (
            SparseTensor,
            SparseTensorFType,
        )

        xp = self.xp
        match node:
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
                vals = [self(arg) for arg in args]
                return func(*vals)

            #access a tensor with only indices
            case ein.Access(tns, idxs) if all(isinstance(idx, ein.Index) for idx in idxs):
                assert len(idxs) == len(set(idxs))
                assert self.loops is not None

                #convert named idxs to positional, integer indices
                perm = [idxs.index(idx) for idx in self.loops if idx in idxs]
                
                tns = self(tns) #evaluate the tensor
                tns = xp.permute_dims(tns, perm) #permute the dimensions
                return xp.expand_dims(
                    tns,
                    [i for i in range(len(self.loops)) if self.loops[i] not in idxs],
                )

            #access a tensor with only one indirect access index
            case ein.Access(tns, idxs):
                assert len(idxs) == 1
                true_idx = node.get_idxs()[0]
                assert isinstance(true_idx, ein.Index)

                raise NotImplementedError("Access with only one indirect access index is not implemented")

            #access a tensor with a mixture of indices and other expressions
            case ein.Access(tns, idxs):
                true_idxs = node.get_idxs() #true field iteratior indicies
                assert all(isinstance(idx, ein.Index) for idx in true_idxs)
                assert self.loops is not None

                raise NotImplementedError("Access with a mixture of indices and other expressions is not implemented")
            case ein.Plan(bodies):
                res = None
                for body in bodies:
                    res = self(body)
                return res
            case ein.Produces(args):
                return tuple(self(arg) for arg in args)

            #get non-zero elements/data array of a sparse tensor
            case ein.GetAttribute(obj, ein.Literal("elems"), _):
                obj = self(obj)
                assert isinstance(ftype(obj), SparseTensorFType)
                assert isinstance(obj, SparseTensor)
                return obj.data 
            #get coord array of a sparse tensor
            case ein.GetAttribute(obj, ein.Literal("coords"), dim):
                obj = self(obj)
                assert isinstance(ftype(obj), SparseTensorFType)
                assert isinstance(obj, SparseTensor)

                # return the coord array for the given dimension or all dimensions
                return obj.coords if dim is None else obj.coords[dim, :]
            case ein.Einsum(op, ein.Alias(tns), idxs, arg):
                # This is the main entry point for einsum execution
                loops = arg.get_idxs()
                
                assert set(idxs).issubset(loops)
                loops = sorted(loops, key=lambda x: x.name)
                ctx = EinsumInterpreter(self.xp, self.bindings, loops)
                arg = ctx(arg)
                axis = tuple(i for i in range(len(loops)) if loops[i] not in idxs)
                op = self(op)
                if op != overwrite:
                    op = getattr(xp, reduction_ops[op])
                    val = op(arg, axis=axis)
                else:
                    assert set(idxs) == set(loops)
                    val = arg
                dropped = [idx for idx in loops if idx in idxs]
                axis = [dropped.index(idx) for idx in idxs]
                self.bindings[tns] = xp.permute_dims(val, axis)
                return (tns,)
            case _:
                raise ValueError(f"Unknown einsum type: {type(node)}")
