import operator

import numpy as np

from finchlite.algebra.tensor import TensorFType
from finchlite.finch_assembly.stages import AssemblyKernel, AssemblyLibrary
from finchlite.finch_einsum.stages import (
    EinsumEvaluator,
    EinsumLoader,
    compute_shape_vars,
)
from finchlite.symbolic.ftype import fisinstance

from ..algebra import overwrite, promote_max, promote_min
from ..symbolic import ftype
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


class PointwiseEinsumMachine:
    def __init__(self, xp, bindings, loops, verbose):
        self.xp = xp
        self.bindings = bindings
        self.loops = loops
        self.verbose = verbose

    def __call__(self, node):
        from ..tensor import (
            SparseTensor,
            SparseTensorFType,
        )

        xp = self.xp
        match node:
            case ein.Literal(val):
                # If val is already an array, return it directly
                # (e.g., from recursive access).
                # Otherwise, broadcast the scalar to match the loop dimensions
                if hasattr(val, "ndim") and val.ndim > 0:
                    return val
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

            # access a tensor with only one indirect access index
            case ein.Access(tns, idxs) if len(idxs) == 1 and not isinstance(
                idxs[0], ein.Index
            ):
                idx = self(idxs[0])
                tns = self(tns)  # evaluate the tensor

                flat_idx = (
                    idx if idx.ndim == 1 else xp.ravel_multi_index(idx.T, tns.shape)
                )
                return tns.flat[flat_idx]  # return a 1-d array by definition

            # access a tensor with an indirect access index
            case ein.Access(tns, idxs) if any(
                not isinstance(idx, ein.Index) for idx in idxs
            ):
                assert self.loops is not None

                tns = self(tns)
                indirect_idxs = [idx for idx in idxs if not isinstance(idx, ein.Index)]

                # iterator indices of the first indirect access
                iterator_idxs = indirect_idxs[0].get_idxs()
                assert len(iterator_idxs) == 1

                # get all indices that share the same iterator
                target_axes = [
                    i
                    for i, idx in enumerate(idxs)
                    if idx.get_idxs().issubset(iterator_idxs)
                ]

                # get associated access indices w/ the first indirect access
                current_idxs = [idxs[i] for i in target_axes]

                # Find the first indirect access to get the iterator size
                first_indirect = indirect_idxs[0]
                indirect_result = self(first_indirect).flat
                iterator_size = len(indirect_result)

                # evaluate the associated access indices
                evaled_idxs: list[np.ndarray] = [
                    xp.arange(iterator_size)
                    if isinstance(idx, ein.Index)
                    else (indirect_result if idx is first_indirect else self(idx).flat)
                    for idx in current_idxs
                ]

                dest_axes = list(range(len(current_idxs)))
                tns = xp.moveaxis(tns, target_axes, dest_axes)

                # access the tensor with the evaled idxs
                tns = tns[tuple(evaled_idxs)]

                # restore original tensor axis order
                # Use min of target_axes since we now include all matching axes
                tns = xp.moveaxis(tns, source=0, destination=target_axes[0])

                # we recursively call the interpreter with the remaining idxs
                [iterator_idx] = iterator_idxs
                # Build new_idxs: replace all current_idxs with iterator_idx at
                # the position of the first target axis
                new_idxs = []
                iterator_placed = False
                j = 0
                for idx in idxs:
                    if j < len(current_idxs) and idx == current_idxs[j]:
                        if not iterator_placed:
                            new_idxs.append(iterator_idx)
                            iterator_placed = True
                        # skip other current_idxs
                        j += 1
                    else:
                        new_idxs.append(idx)

                new_access = ein.Access(ein.Literal(tns), new_idxs)
                return self(new_access)
            # access a tensor with a mixture of indices and other expressions
            case ein.Access(tns, idxs):
                assert self.loops is not None

                tns = self(tns)
                perm = [idxs.index(idx) for idx in self.loops if idx in idxs]
                if hasattr(tns, "ndim") and len(perm) < tns.ndim:
                    perm += list(range(len(perm), tns.ndim))

                tns = xp.permute_dims(tns, perm)  # permute the dimensions
                return xp.expand_dims(
                    tns,
                    [i for i in range(len(self.loops)) if self.loops[i] not in idxs],
                )
            # get non-zero elements/data array of a sparse tensor
            case ein.GetAttr(obj, ein.Literal("elems"), _):
                obj = self(obj)
                assert isinstance(obj, SparseTensor)
                return obj.data
            # get coord array of a sparse tensor
            case ein.GetAttr(obj, ein.Literal("coords"), dim):
                obj = self(obj)
                assert isinstance(ftype(obj), SparseTensorFType)
                assert isinstance(obj, SparseTensor)

                # return the coord array for the given dimension or all dimensions
                return obj.coords if dim is None else obj.coords[:, dim].flat
            # gets the shape of a sparse tensor at a given dimension
            case ein.GetAttr(obj, ein.Literal("shape"), dim):
                obj = self(obj)
                assert isinstance(ftype(obj), SparseTensorFType)
                assert isinstance(obj, SparseTensor)
                assert dim is not None

                # return the shape for the given dimension
                return obj.shape[dim]
            case _:
                raise ValueError(f"Unknown einsum type: {type(node)}")


class EinsumMachine:
    def __init__(self, xp, bindings, verbose):
        self.xp = xp
        self.bindings = bindings
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
                ctx = PointwiseEinsumMachine(
                    self.xp, self.bindings, loops, self.verbose
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
