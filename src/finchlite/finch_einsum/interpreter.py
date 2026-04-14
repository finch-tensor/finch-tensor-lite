import numpy as np
from dataclasses import dataclass
from typing import Any

from finchlite.algebra.tensor import TensorFType
from finchlite.finch_assembly.stages import AssemblyKernel, AssemblyLibrary
from finchlite.finch_einsum.stages import (
    EinsumEvaluator,
    EinsumLoader,
    compute_shape_vars,
)
from finchlite.symbolic.ftype import fisinstance

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
    ffunc.promote_min: "minimum",
    ffunc.promote_max: "maximum",
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
    ffunc.promote_min: "min",
    ffunc.promote_max: "max",
    ffunc.logical_and: "all",
    ffunc.logical_or: "any",
}


@dataclass
class DistributedFinchTensor:
    local: Any
    communicator: "Collective"
    partition: str = "dense_block"
    global_shape: tuple[int, ...] | None = None

    def __post_init__(self):
        self.local = np.asarray(self.local)
        if self.global_shape is None:
            self.global_shape = tuple(self.local.shape)


class Collective:
    def __init__(self, rank: int = 0, size: int = 1):
        self.rank = rank
        self.size = size
        self._agreed_kernel: str | None = None
        self._mailbox: dict[tuple[int, int, str], Any] = {}

    def send(self, value: Any, dst: int, tag: str = "default"):
        self._mailbox[(self.rank, dst, tag)] = value

    def recv(self, src: int, tag: str = "default") -> Any:
        key = (src, self.rank, tag)
        if key not in self._mailbox:
            raise ValueError(f"No message available for src={src}, tag={tag}")
        return self._mailbox.pop(key)

    def broadcast(self, value: Any, root: int = 0) -> Any:
        return value

    def allreduce_sum(self, value: Any) -> Any:
        return value

    def run_same_kernel(self, kernel_token: str):
        if self._agreed_kernel is None:
            self._agreed_kernel = kernel_token
        elif self._agreed_kernel != kernel_token:
            raise ValueError(
                f"Ranks disagreed on kernel: {self._agreed_kernel} != {kernel_token}"
            )


class SummaEinsumKernel(AssemblyKernel):
    def __init__(self, prgm, bindings: dict[ein.Alias, TensorFType], collective: Collective, block_k: int):
        self.prgm = prgm
        self.bindings = bindings
        self.collective = collective
        self.block_k = block_k

    def __call__(self, *args):
        if len(args) != len(self.bindings):
            raise ValueError(
                f"Wrong number of arguments passed to kernel, "
                f"have {len(args)}, expected {len(self.bindings)}"
            )

        b = dict(zip(self.bindings.keys(), args, strict=True))
        for k, v in b.items():
            if not isinstance(v, DistributedFinchTensor):
                assert fisinstance(v, self.bindings[k])

        match self.prgm:
            case ein.Einsum(
                ein.Literal(reduce_op),
                out,
                (i, j),
                ein.Call(
                    ein.Literal(mul_op),
                    (
                        ein.Access(lhs, (i2, k1)),
                        ein.Access(rhs, (k2, j2)),
                    ),
                ),
            ) if (
                mul_op == ffunc.mul
                and i == i2
                and j == j2
                and k1 == k2
                and reduce_op in (ffunc.overwrite, ffunc.add)
            ):
                pass
            case ein.Plan(bodies):
                stmt = next((x for x in bodies if isinstance(x, ein.Einsum)), None)
                if stmt is None:
                    raise ValueError("SUMMA kernel expects a matrix-multiply einsum")
                match stmt:
                    case ein.Einsum(
                        ein.Literal(reduce_op),
                        out,
                        (i, j),
                        ein.Call(
                            ein.Literal(mul_op),
                            (
                                ein.Access(lhs, (i2, k1)),
                                ein.Access(rhs, (k2, j2)),
                            ),
                        ),
                    ) if (
                        mul_op == ffunc.mul
                        and i == i2
                        and j == j2
                        and k1 == k2
                        and reduce_op in (ffunc.overwrite, ffunc.add)
                    ):
                        pass
                    case _:
                        raise ValueError("SUMMA kernel expects a matrix-multiply einsum")
            case _:
                raise ValueError("SUMMA kernel expects a matrix-multiply einsum")

        self.collective.run_same_kernel(f"summa_mvp::{out.name}={lhs.name}@{rhs.name}")
        a_obj, r_obj = b[lhs], b[rhs]
        a = np.asarray(a_obj.local if isinstance(a_obj, DistributedFinchTensor) else a_obj)
        r = np.asarray(r_obj.local if isinstance(r_obj, DistributedFinchTensor) else r_obj)
        if a.ndim != 2 or r.ndim != 2:
            raise ValueError("SUMMA MVP supports rank-2 matrix multiplication only")
        if a.shape[1] != r.shape[0]:
            raise ValueError(f"Invalid matmul shapes: {a.shape} x {r.shape}")

        m, kdim = a.shape
        _, n = r.shape
        bk = max(1, min(self.block_k, kdim))
        result = np.zeros((m, n), dtype=np.result_type(a.dtype, r.dtype))
        for k0 in range(0, kdim, bk):
            kx = min(k0 + bk, kdim)
            pa = self.collective.broadcast(a[:, k0:kx], root=0)
            pb = self.collective.broadcast(r[k0:kx, :], root=0)
            result += pa @ pb
        result = np.asarray(self.collective.allreduce_sum(result))

        if reduce_op == ffunc.add:
            o = b[out]
            o_arr = np.asarray(o.local if isinstance(o, DistributedFinchTensor) else o)
            result = o_arr + result

        o = b[out]
        if isinstance(o, DistributedFinchTensor):
            o.local[...] = result
        elif hasattr(o, "shape") and hasattr(o, "dtype"):
            o[...] = result
        else:
            o = result
        b[out] = o
        return (b[out],)


@dataclass
class SummaEinsumLibrary(AssemblyLibrary):
    prgm: Any
    bindings: dict[ein.Alias, TensorFType]
    collective: Collective
    block_k: int

    def __getattr__(self, name):
        if name == "main":
            return SummaEinsumKernel(self.prgm, self.bindings, self.collective, self.block_k)
        if name == "prgm":
            return self.prgm
        raise AttributeError(f"Unknown attribute {name} for SummaEinsumLibrary")


class SummaEinsumLoader(EinsumLoader):
    def __init__(self, collective: Collective | None = None, block_k: int = 64):
        self.collective = collective or Collective()
        self.block_k = block_k

    def __call__(self, prgm: ein.EinsumStatement, bindings: dict[ein.Alias, TensorFType]):
        shape_vars = compute_shape_vars(prgm, bindings)
        return (SummaEinsumLibrary(prgm, bindings, self.collective, self.block_k), bindings, shape_vars)


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
