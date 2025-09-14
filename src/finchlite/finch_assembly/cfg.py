from .nodes import (
    AssemblyNode,
    Assign,
    Block,
    Break,
    BufferLoop,
    Call,
    ForLoop,
    Function,
    GetAttr,
    If,
    IfElse,
    Length,
    Literal,
    Load,
    Module,
    Repack,
    Resize,
    Return,
    SetAttr,
    Slot,
    Stack,
    Store,
    Unpack,
    Variable,
    WhileLoop,
)


class BasicBlock:
    def __init__(self, id):
        self.id = id
        self.statements = []
        self.successors = []
        self.predecessors = []

    def add_statement(self, statement):
        self.statements.append(statement)

    def add_successor(self, succ_id: str, blocks: dict[str, "BasicBlock"]) -> None:
        if succ_id not in self.successors:
            self.successors.append(succ_id)

        if self.id not in blocks[succ_id].predecessors:
            blocks[succ_id].predecessors.append(self.id)

    def __repr__(self):
        return (
            f"BasicBlock(id={self.id}, stmts={self.statements}, "
            f"succs={self.successors})"
        )


class CFG:
    def __init__(self, func_name: str):
        self.block_counter = 0
        self.name = func_name
        self.blocks = {}

        # initialize ENTRY and EXIT blocks
        self.entry_block = self.new_block()
        self.exit_block = self.new_block()

        self.current_block = self.new_block()
        self.entry_block.add_successor(self.current_block, self.blocks)

    def new_block(self):
        bid = f"{self.name}_{self.block_counter}"
        self.block_counter += 1
        block = BasicBlock(bid)
        self.blocks[bid] = block
        return block


class CFGBuilder:  # is it required to have a 'main' entry point for a program?
    def __init__(self):
        self.cfgs = {}
        self.current_cfg = None

    def new_cfg(self, name: str) -> CFG:
        new_cfg = CFG(name)
        self.cfgs[name] = new_cfg
        return new_cfg

    def build(self, node: AssemblyNode):
        return self(node)

    def __call__(self, node: AssemblyNode, break_block_id: str = None):
        match node:
            case Literal(value):
                self.current_cfg.current_block.add_statement(("literal", value))
            case Unpack(lhs, rhs):
                self.current_cfg.current_block.add_statement(("unpack", lhs, rhs))
            case Repack(val):
                self.current_cfg.current_block.add_statement(("repack", val))
            case Resize(buffer, new_size):
                self.current_cfg.current_block.add_statement(
                    ("resize", buffer, new_size)
                )
            case Variable(name, type):
                self.current_cfg.current_block.add_statement(("variable", name, type))
            case GetAttr(obj, attr):
                self.current_cfg.current_block.add_statement(("getattr", obj, attr))
            case SetAttr(obj, attr, value):
                self.current_cfg.current_block.add_statement(
                    ("setattr", obj, attr, value)
                )
            case Call(Literal(_) as lit, args):
                self.current_cfg.current_block.add_statement(
                    ("function_call", lit, args)
                )
            case Load(buffer, index):
                self.current_cfg.current_block.add_statement(("load", buffer, index))
            case Store(buffer, index, value):
                self.current_cfg.current_block.add_statement(
                    ("store", buffer, index, value)
                )
            case Length(buffer):
                self.current_cfg.current_block.add_statement(("length", buffer))
            case Slot(name, type):
                self.current_cfg.current_block.add_statement(("slot", name, type))
            case Stack(obj, type):
                self.current_cfg.current_block.add_statement(("stack", obj, type))
            case Assign(Variable(name, _), val):
                self.current_cfg.current_block.add_statement(("assign", name, val))
            case Block(bodies):
                for body in bodies:
                    self(body, break_block_id)
            case If(cond, body):
                cond_block = self.current_cfg.current_block
                cond_block.add_statement(("if_cond", cond))

                then_block = self.current_cfg.new_block()
                self.current_cfg.current_block = then_block
                self(body, break_block_id)

                after_block = self.current_cfg.new_block()
                cond_block.add_successor(then_block.id, self.current_cfg.blocks)
                cond_block.add_successor(after_block.id, self.current_cfg.blocks)

                self.current_cfg.current_block.add_successor(
                    after_block.id, self.current_cfg.blocks
                )
                self.current_cfg.current_block = after_block
            case IfElse(cond, body, else_body):
                cond_block = self.current_cfg.current_block
                cond_block.add_statement(("if_else_cond", cond))

                then_block = self.current_cfg.new_block()
                self.current_cfg.current_block = then_block
                self(body, break_block_id)

                after_block = self.current_cfg.new_block()
                cond_block.add_successor(then_block.id, self.current_cfg.blocks)
                cond_block.add_successor(after_block.id, self.current_cfg.blocks)
                self.current_cfg.current_block.add_successor(
                    after_block.id, self.current_cfg.blocks
                )

                else_block = self.current_cfg.new_block()
                self.current_cfg.current_block = else_block
                self(else_body, break_block_id)

                self.current_cfg.current_block.add_successor(
                    after_block.id, self.current_cfg.blocks
                )
                self.current_cfg.current_block = after_block
            case WhileLoop(cond, body):
                cond_block = self.current_cfg.current_block
                cond_block.add_statement(("while_cond", cond))

                body_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                cond_block.add_successor(body_block.id, self.current_cfg.blocks)
                cond_block.add_successor(after_block.id, self.current_cfg.blocks)

                self.current_cfg.current_block = body_block
                self(body, after_block.id)

                self.current_cfg.current_block.add_successor(
                    cond_block.id, self.current_cfg.blocks
                )
                self.current_cfg.current_block = after_block
            case ForLoop(var, start, end, body):
                init_block = self.current_cfg.current_block
                init_block.add_statement(("for_init", var, start, end))

                cond_block = self.current_cfg.new_block()
                init_block.add_successor(cond_block.id, self.current_cfg.blocks)
                cond_block.add_statement(("for_init", var, start, end))

                body_block = self.current_cfg.new_block()
                cond_block.add_successor(body_block, self.current_cfg.blocks)

                after_block = self.current_cfg.new_block()
                cond_block.add_successor(after_block, self.current_cfg.blocks)

                self.current_cfg.current_block = body_block
                self(body_block, after_block.id)

                self.current_cfg.current_block.add_statement(("for_inc", var))
                self.current_cfg.current_block.add_successor(
                    cond_block.id, self.current_cfg.blocks
                )

                self.current_cfg.current_block = after_block
            case BufferLoop(buf, var, body):
                init_block = self.current_cfg.current_block
                init_block.add_statement(("bufferloop_init", buf, var))

                cond_block = self.current_cfg.new_block()
                init_block.add_successor(cond_block.id, self.current_cfg.blocks)
                cond_block.add_statement(("bufferloop_init", var, start, end))

                body_block = self.current_cfg.new_block()
                cond_block.add_successor(body_block, self.current_cfg.blocks)

                after_block = self.current_cfg.new_block()
                cond_block.add_successor(after_block, self.current_cfg.blocks)

                self.current_cfg.current_block = body_block
                self(body_block, after_block.id)

                self.current_cfg.current_block.add_statement(("bufferloop_inc", var))
                self.current_cfg.current_block.add_successor(
                    cond_block.id, self.current_cfg.blocks
                )

                self.current_cfg.current_block = after_block
            case Return(value):
                self.current_cfg.current_block.add_statement(("return", value))

                # when Return is met,
                # make a connection to the EXIT block of function (cfg)
                self.current_cfg.current_block.add_successor(
                    self.current_cfg.exit_block.id, self.current_cfg.blocks
                )

                # create a block where we going to store all unreachable statements
                unreachable_block = self.current_cfg.new_block()
                self.current_cfg.current_block = unreachable_block
            case Break():
                self.current_cfg.current_block.add_statement("break")

                # when Break is met,
                # make a connection to the AFTER block of ForLoop/WhileLoop
                self.current_cfg.current_block.add_successor(
                    break_block_id, self.current_cfg.blocks
                )

                # create a block where we going to store all unreachable statements
                unreachable_block = self.current_cfg.new_block()
                self.current_cfg.current_block = unreachable_block
            case Function(Variable(_, _), args, body):
                for arg in args:
                    match arg:
                        case Variable(name, type):
                            self.current_cfg.current_block.add_statement(
                                ("func_arg"), name, type
                            )
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )

                self(body)
            case Module(funcs):
                for func in funcs:
                    if not isinstance(func, Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )

                    self.current_cfg = self.new_cfg(func.name.name)
                    self(func)
                    end_func_block = self.current_cfg.new_block()
                    self.current_cfg.current_block.add_successor(
                        end_func_block, self.current_cfg.blocks
                    )
                    self.current_cfg.current_block = end_func_block
            case node:
                raise NotImplementedError(node)

        return self.cfgs


"""
cfgs: {
    main: {
        main_0: {...statements},
        main_1: {...statements},
        main_2: {...statements},
    },
    func: {
        func_0: {...statements},
        func_1: {...statements},
    },
    .
    .
    .
}
"""


# TODO: CFG Printer
class CFGPrinter: ...


# TODO: create TestClass to test the CFG builder
class CFGTest: ...
