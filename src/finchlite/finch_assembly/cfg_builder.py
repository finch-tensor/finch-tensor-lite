import operator

from ..dataflow.cfg import BasicBlock, ControlFlowGraph
from .nodes import (
    AssemblyNode,
    Assert,
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
    TaggedVariable,
    Unpack,
    WhileLoop,
)


class CFGBuilder:
    """Incrementally builds control-flow graphs for Finch Assembly IR."""

    def __init__(self):
        self.cfgs: dict[str, ControlFlowGraph] = {}
        self.current_block = None
        self.current_cfg: ControlFlowGraph

    def new_cfg(self, name: str) -> ControlFlowGraph:
        new_cfg = ControlFlowGraph(name)
        self.current_block = new_cfg.new_block()
        new_cfg.entry_block.add_successor(self.current_block)
        self.cfgs[name] = new_cfg
        return new_cfg

    def build(self, node: AssemblyNode):
        return self(node)

    def __call__(self, node: AssemblyNode, break_block: BasicBlock | None = None):
        match node:
            case (
                Literal()
                | Unpack()
                | Repack()
                | Resize()
                | TaggedVariable()
                | GetAttr()
                | SetAttr()
                | Call()
                | Load()
                | Store()
                | Length()
                | Slot()
                | Stack()
                | Assign()
                | Assert()
            ):
                self.current_block.add_statement(node)
            case Block(bodies):
                for body in bodies:
                    self(body, break_block)
            case If(cond, body):
                self(IfElse(cond, body, Block()))
            case IfElse(cond, body, else_body):
                before_block = self.current_block

                if_block = self.current_cfg.new_block()
                else_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                before_block.add_successor(if_block)
                before_block.add_successor(else_block)

                self.current_block = if_block
                self.current_block.add_statement(Assert(cond))
                self(body, break_block)
                self.current_block.add_successor(after_block)

                self.current_block = else_block
                self.current_block.add_statement(
                    Assert(
                        Call(
                            Literal(operator.not_),
                            (cond,),
                        )
                    )
                )
                self(else_body, break_block)
                self.current_block.add_successor(after_block)

                self.current_block = after_block
            case WhileLoop(cond, body):
                before_block = self.current_block

                body_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                before_block.add_successor(body_block)
                before_block.add_successor(after_block)

                self.current_block = body_block
                self.current_block.add_statement(Assert(cond))
                self(body, after_block)

                self.current_block.add_successor(before_block)
                self.current_block = after_block
            case ForLoop(var, _start, _end, body):
                before_block = self.current_block

                body_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                before_block.add_successor(body_block)
                before_block.add_successor(after_block)

                # TODO: figure out a RIGHT way to represent 'ForLoop' initialization
                # statement
                body_block.add_statement(Assign(var, var))
                self.current_block = body_block
                self(body, after_block)

                self.current_block.add_successor(body_block)
                self.current_block = after_block
            case BufferLoop(_buf, var, body):
                before_block = self.current_block

                body_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                before_block.add_successor(body_block)
                before_block.add_successor(after_block)

                # TODO: figure out a RIGHT way to represent 'BufferLoop' initialization
                # statement
                body_block.add_statement(Assign(var, var))
                self.current_block = body_block
                self(body, after_block)

                self.current_block.add_successor(body_block)
                self.current_block = after_block
            case Return(value):
                self.current_block.add_statement(Return(value))

                # when Return is met,
                # make a connection to the EXIT block of function (cfg)
                self.current_block.add_successor(self.current_cfg.exit_block)

                # create a block where we going to store all unreachable statements
                unreachable_block = self.current_cfg.new_block()
                self.current_block = unreachable_block
            case Break():
                self.current_block.add_statement(Break())
                self.current_block.add_successor(break_block)
                unreachable_block = self.current_cfg.new_block()
                self.current_block = unreachable_block
            case Function(_, args, body):
                for arg in args:
                    match arg:
                        case TaggedVariable():
                            self.current_block.add_statement(arg)
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

                    func_name = (
                        func.name.variable.name
                        if hasattr(func.name, "variable")
                        else func.name.name
                    )

                    self.current_cfg = self.new_cfg(func_name)
                    self(func)
                    self.current_block.add_successor(self.current_cfg.exit_block)
            case node:
                raise NotImplementedError(node)

        return self.cfgs
