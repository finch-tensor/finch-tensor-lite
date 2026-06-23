from finchlite import finch_assembly as asm
from finchlite.symbolic import Context, PostOrderDFS, ScopedDict


def loop_vars(node: asm.AssemblyNode):
    out: set[str] = set()
    for n in PostOrderDFS(node):
        match n:
            case asm.Assign(asm.Variable(name, _), _):
                out.add(name)
    return out


class MLIRContext(Context):
    def __init__(
        self,
        tab="  ",
        indent=1,
        bindings=None,
    ):
        if bindings is None:
            bindings = ScopedDict()

        super().__init__()
        self.tab = tab
        self.indent = indent
        self.bindings = bindings

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def emit_global(self):
        return "\n".join([*self.headers, self.emit()])

    def block(self) -> "MLIRContext":
        blk = super().block()
        blk.tab = self.tab
        blk.indent = self.indent
        blk.bindings = self.bindings
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        blk.bindings = self.bindings.scope()
        return blk

    def __call__(self, prgm: asm.AssemblyNode):
        # feed = self.feed
        match prgm:
            # case asm.Literal(value):
            #     ...

            # case asm.Variable(name, _):
            #     ...

            # case asm.Assign(asm.Variable(var_n, var_t) as var, val):
            #     ...

            # case asm.Call(op, args):
            #     ...

            # case asm.Load(buffer, index):
            #     ...

            # case asm.Store(buffer, index, value):
            #     ...

            # case asm.Block(bodies):
            #     ...

            # case asm.ForLoop(asm.Variable(var_n, var_t), start, end, body):
            #     ...

            # case asm.WhileLoop(cond, body):
            #     ...

            # case asm.BufferLoop(buf, var, body):
            #     ...

            # case asm.If(cond, body):
            #     ...

            # case asm.IfElse(cond, body, else_body):
            #     ...

            # case asm.Function(asm.Variable(func_name, return_t), args, body):
            #     ...

            # case asm.Return(value):
            #     if value.result_type == algebra.none_:
            #         self.exec(f"{feed}func.return")

            case asm.Module(funcs):
                for func in funcs:
                    if not isinstance(func, asm.Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func)
                return

            case node:
                raise NotImplementedError(f"Unrecognized node: {node}")
