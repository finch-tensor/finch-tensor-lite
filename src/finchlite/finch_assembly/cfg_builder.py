import operator

import numpy as np

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
    Variable,
    WhileLoop,
)


class CFGBuilder:
    """Incrementally builds control-flow graphs for Finch Assembly IR."""

    def __init__(self):
        self.cfgs: dict[str, ControlFlowGraph] = {}
        self.current_block = None
        self.current_cfg: ControlFlowGraph
        self.loop_counter_id = 0

    def new_cfg(self, name: str) -> ControlFlowGraph:
        new_cfg = ControlFlowGraph(name)
        self.current_block = new_cfg.new_block()
        new_cfg.entry_block.add_successor(self.current_block)
        self.cfgs[name] = new_cfg
        return new_cfg

    def get_loop_counter_id(self):
        current_id = self.loop_counter_id
        self.loop_counter_id += 1
        return current_id

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
                self(IfElse(cond, body, Block()), break_block)
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
            case ForLoop(var, start, end, body):
                before_block = self.current_block

                # create fictitious variable
                fic_var = TaggedVariable(
                    Variable("for_loop_counter", np.int64), self.get_loop_counter_id()
                )
                before_block.add_statement(Assign(fic_var, start))

                # create while loop condition: j < end
                loop_condition = Call(Literal(operator.lt), (fic_var, end))

                # create loop body with i = j assignment and increment
                loop_body = Block(
                    (
                        Assign(var, fic_var),
                        body,
                        Assign(
                            fic_var,
                            Call(
                                Literal(operator.add),
                                (fic_var, Literal(np.int64(1))),
                            ),
                        ),
                    )
                )

                self(WhileLoop(loop_condition, loop_body), break_block)
            case BufferLoop(buf, var, body):
                before_block = self.current_block

                fic_var = TaggedVariable(
                    Variable("buffer_loop_counter", np.int64),
                    self.get_loop_counter_id(),
                )
                before_block.add_statement(Assign(fic_var, Literal(np.int64(0))))

                # create while loop condition: i < length(buf)
                loop_condition = Call(Literal(operator.lt), (fic_var, Length(buf)))

                # create loop body with var = buf[i] assignment and increment
                loop_body = Block(
                    (
                        Assign(var, Load(buf, fic_var)),
                        body,
                        Assign(
                            fic_var,
                            Call(
                                Literal(operator.add),
                                (fic_var, Literal(np.int64(1))),
                            ),
                        ),
                    )
                )

                self(WhileLoop(loop_condition, loop_body), break_block)
            case Return(value):
                self.current_block.add_statement(Return(value))
                self.current_block.add_successor(self.current_cfg.exit_block)
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

                    if isinstance(func.name, TaggedVariable):
                        func_name = func.name.variable.name
                    elif isinstance(func.name, Variable):
                        func_name = func.name.name
                    else:
                        raise NotImplementedError(
                            f"Unrecognized function name type: {type(func.name)}"
                        )

                    self.current_cfg = self.new_cfg(func_name)
                    self(func)
                    self.current_block.add_successor(self.current_cfg.exit_block)
            case node:
                raise NotImplementedError(node)

        return self.cfgs
