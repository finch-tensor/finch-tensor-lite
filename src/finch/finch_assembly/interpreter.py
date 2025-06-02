from __future__ import annotations
from ..symbolic import ScopedDict
from . import nodes as asm

class AssemblyInterpreterKernel():
    """
    A kernel for interpreting FinchAssembly code.
    This is a simple interpreter that executes the assembly code.
    """
    def __init__(self, prgm, func_n, ret_t):
        self.ctx = AssemblyInterpreter()
        self.func = asm.Variable(self.func_n, ret_t)
        self.ctx(prgm)

    def __call__(self, *args):
        args_i = (asm.Immediate(arg) for arg in args)
        return self.ctx(asm.Call(self.func, args_i))

class AssemblyInterpreter():
    """
    An interpreter for FinchAssembly.
    """
    def __init__(self, bindings=None, types=None, loop = None, ret=None):
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


    def scope(self, **kwargs):
        """
        Create a new scope for the interpreter.
        This allows for nested scopes and variable shadowing.
        """
        return AssemblyInterpreter(
            bindings=self.bindings.scope(),
            types=self.types.scope(),
            loop=self.loop,
            ret=self.ret,
            **kwargs
        )


    def should_halt(self):
        """
        Check if the interpreter should halt execution.
        This is used to stop execution in loops or when a return statement is encountered.
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
                    var_e = self.bindings[var_n]
                    return var_e
                raise KeyError(
                    f"Variable '{var_n}' is not defined in the current context."
                )
            case asm.Assign(asm.Variable(var_n, var_t), val):
                val_e = self(val)
                if not isinstance(val_e, var_t):
                    raise TypeError(
                        f"Assigned value {val_e} is not of type {var_t} for variable '{var}'."
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
                return buf.load(buf_e, idx_e)
            case asm.Store(buf, idx, val):
                buf_e = self(buf)
                idx_e = self(idx)
                val_e = self(val)
                buf_e.store(idx_e, val_e)
                return None
            case asm.Resize(buf, len):
                buf_e = self(buf)
                len_e = self(len)
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
                        f"Start value {start_e} is not of type {var_t} for variable '{var_n}'."
                    )
                if not isinstance(end_e, var_t):
                    raise TypeError(
                        f"End value {end_e} is not of type {var_t} for variable '{var_n}'."
                    )
                ctx_2 = self.scope(loop=[])
                for var_e in range(start_e, end_e):
                    if ctx_2.should_halt():
                        break
                    ctx_3 = self.scope()
                    ctx_3(asm.Block(
                        asm.Assign(var, asm.Immediate(var_e)),
                        body
                    ))
                return None
            case asm.BufferLoop(buf, var, body):
                ctx_2 = self.scope(loop=[])
                for i in range(buf.length()):
                    if ctx_2.should_halt():
                        break
                    ctx_3 = ctx_2.scope()
                    ctx_3(asm.Block(
                        asm.Assign(var, asm.Load(buf, asm.Immediate(i))),
                        body
                    ))
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
                    ctx_2 = self.scope(ret = [])
                    if len(args_e) != len(args):
                        raise ValueError(
                            f"Function '{func_n}' expects {len(args)} arguments, "
                            f"but got {len(args_e)}."
                        )
                    for arg, arg_e in zip(args, args_e):
                        match arg:
                            case asm.Variable(arg_n, arg_t):
                                if not isinstance(arg_e, arg_t):
                                    raise TypeError(
                                        f"Argument '{arg_n}' is expected to be of type {arg_t}, "
                                        f"but got {type(arg_e)}."
                                    )
                                ctx_2.bindings[arg_n] = arg_e
                            case _:
                                raise NotImplementedError(
                                    f"Unrecognized argument type: {arg}"
                                )
                    ctx_2(body)
                    if len(ctx_2.ret) > 0:
                        ret_e = ctx_2.ret[1]
                        if not isinstance(ret_e, ret_t):
                            raise TypeError(
                                f"Return value {ret_e} is not of type {ret_t} for function '{func_n}'."
                            )
                        return ret_e
                    else:
                        raise ValueError(
                            f"Function '{func_n}' did not return a value, "
                            f"but expected type {ret_t}."
                        )
                self.bindings[func_n] = my_func
            case asm.Return(value):
                self.ret.append(self(value))
                return None
            case asm.Break():
                self.loop.append([])
                return None
