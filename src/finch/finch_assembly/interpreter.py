from __future__ import annotations

from ..symbolic import ScopedDict
from . import nodes as asm
from .abstract_buffer import isinstanceorformat


class AssemblyInterpreterKernel:
    """
    A kernel for interpreting FinchAssembly code.
    This is a simple interpreter that executes the assembly code.
    """

    def __init__(self, ctx, func_n, ret_t):
        self.ctx = ctx
        self.func = asm.Variable(func_n, ret_t)

    def __call__(self, *args):
        args_i = (asm.Immediate(arg) for arg in args)
        return self.ctx(asm.Call(self.func, args_i))


class AssemblyInterpreterModule:
    """
    A class to represent an interpreted module of FinchAssembly.
    """

    def __init__(self, ctx, kernels):
        self.ctx = ctx
        self.kernels = kernels

    def __getattr__(self, name):
        # Allow attribute access to kernels by name
        if name in self.kernels:
            return self.kernels[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class AssemblyInterpreter:
    """
    An interpreter for FinchAssembly.
    """

    def __init__(self, bindings=None, types=None, loop=None, ret=None):
        if bindings is None:
            bindings = ScopedDict()
        if types is None:
            types = ScopedDict()
        if loop is None:
            loop = []
        if ret is None:
            ret = []
        self.bindings = bindings
        self.types = types
        self.loop = loop
        self.ret = ret

    def scope(self, bindings=None, types=None, loop=None, ret=None):
        """
        Create a new scope for the interpreter.
        This allows for nested scopes and variable shadowing.
        """
        if bindings is None:
            bindings = self.bindings.scope()
        if types is None:
            types = self.types.scope()
        if loop is None:
            loop = self.loop
        if ret is None:
            ret = self.ret
        return AssemblyInterpreter(
            bindings=bindings,
            types=types,
            loop=loop,
            ret=ret,
        )

    def should_halt(self):
        """
        Check if the interpreter should halt execution.
        This is used to stop execution in loops or when a return
        statement is encountered.
        """
        return self.loop or self.ret

    def __call__(self, prgm: asm.AssemblyNode):
        """
        Run the program.
        """
        match prgm:
            case asm.Immediate(value):
                return value
            case asm.Variable(var_n, var_t):
                if var_n in self.types:
                    def_t = self.types[var_n]
                    if def_t != var_t:
                        raise TypeError(
                            f"Variable '{var_n}' is declared as type {def_t}, "
                            f"but used as type {var_t}."
                        )
                if var_n in self.bindings:
                    return self.bindings[var_n]
                raise KeyError(
                    f"Variable '{var_n}' is not defined in the current context."
                )
            case asm.Assign(asm.Variable(var_n, var_t), val):
                val_e = self(val)
                if not isinstance(val_e, var_t):
                    raise TypeError(
                        f"Assigned value {val_e} is not of type {var_t} for "
                        f"variable '{var_n}'."
                    )
                self.bindings[var_n] = val_e
                self.types[var_n] = var_t
                return None
            case asm.Call(f, args):
                f_e = self(f)
                args_e = [self(arg) for arg in args]
                return f_e(*args_e)
            case asm.Load(buf, idx):
                buf_e = self(buf)
                idx_e = self(idx)
                return buf_e.load(idx_e)
            case asm.Store(buf, idx, val):
                buf_e = self(buf)
                idx_e = self(idx)
                val_e = self(val)
                buf_e.store(idx_e, val_e)
                return None
            case asm.Resize(buf, len_):
                buf_e = self(buf)
                len_e = self(len_)
                buf_e.resize(len_e)
                return None
            case asm.Length(buf):
                buf_e = self(buf)
                return buf_e.length()
            case asm.Block(bodies):
                for body in bodies:
                    if self.should_halt():
                        break
                    self(body)
                return None
            case asm.ForLoop(asm.Variable(var_n, var_t) as var, start, end, body):
                start_e = self(start)
                end_e = self(end)
                if not isinstance(start_e, var_t):
                    raise TypeError(
                        f"Start value {start_e} is not of type {var_t} for "
                        f"variable '{var_n}'."
                    )
                ctx_2 = self.scope(loop=[])
                var_e = start_e
                while var_e < end_e:
                    if ctx_2.should_halt():
                        break
                    ctx_3 = self.scope()
                    ctx_3(asm.Block((asm.Assign(var, asm.Immediate(var_e)), body)))
                    var_e = type(var_e)(var_e + 1)  # type: ignore[call-arg,operator]
                return None
            case asm.BufferLoop(buf, var, body):
                ctx_2 = self.scope(loop=[])
                buf_e = self(buf)
                for i in range(buf_e.length()):
                    if ctx_2.should_halt():
                        break
                    ctx_3 = ctx_2.scope()
                    ctx_3(
                        asm.Block(
                            (asm.Assign(var, asm.Load(buf, asm.Immediate(i))), body)
                        )
                    )
                return None
            case asm.WhileLoop(cond, body):
                ctx_2 = self.scope(loop=[])
                while self(cond):
                    ctx_3 = ctx_2.scope()
                    if ctx_3.should_halt():
                        break
                    ctx_3(body)
                return None
            case asm.Function(asm.Variable(func_n, ret_t), args, body):

                def my_func(*args_e):
                    ctx_2 = self.scope(ret=[])
                    if len(args_e) != len(args):
                        raise ValueError(
                            f"Function '{func_n}' expects {len(args)} arguments, "
                            f"but got {len(args_e)}."
                        )
                    for arg, arg_e in zip(args, args_e, strict=False):
                        match arg:
                            case asm.Variable(arg_n, arg_t):
                                if not isinstanceorformat(arg_e, arg_t):
                                    raise TypeError(
                                        f"Argument '{arg_n}' is expected to be of type "
                                        f"{arg_t}, but got {type(arg_e)}."
                                    )
                                ctx_2.bindings[arg_n] = arg_e
                            case _:
                                raise NotImplementedError(
                                    f"Unrecognized argument type: {arg}"
                                )
                    ctx_2(body)
                    if len(ctx_2.ret) > 0:
                        ret_e = ctx_2.ret[0]
                        if not isinstance(ret_e, ret_t):
                            raise TypeError(
                                f"Return value {ret_e} is not of type {ret_t} "
                                f"for function '{func_n}'."
                            )
                        return ret_e
                    raise ValueError(
                        f"Function '{func_n}' did not return a value, "
                        f"but expected type {ret_t}."
                    )

                self.bindings[func_n] = my_func
                return None
            case asm.Return(value):
                self.ret.append(self(value))
                return None
            case asm.Break():
                self.loop.append([])
                return None
            case asm.Module(funcs):
                for func in funcs:
                    self(func)
                kernels = {}
                for func in funcs:
                    match func:
                        case asm.Function(asm.Variable(func_n, ret_t), args, _):
                            kernel = AssemblyInterpreterKernel(self, func_n, ret_t)
                            kernels[func_n] = kernel
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized function definition: {func}"
                            )
                return AssemblyInterpreterModule(self, kernels)
            case _:
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )
