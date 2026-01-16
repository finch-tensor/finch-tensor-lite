import operator

import numpy as np

from ..symbolic import (
    BasicBlock,
    ControlFlowGraph,
    Namespace,
)
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
    If,
    IfElse,
    Length,
    Literal,
    Load,
    Module,
    Print,
    Repack,
    Resize,
    Return,
    SetAttr,
    Store,
    Unpack,
    Variable,
    WhileLoop,
)


def assembly_build_cfg(node: AssemblyNode):
    ctx = AssemblyCFGBuilder(namespace=Namespace(node))
    return ctx.build(node)


class AssemblyCFGBuilder:
    """Incrementally builds control-flow graph for Finch Assembly IR."""

    def __init__(self, namespace: Namespace | None = None):
        self.cfg: ControlFlowGraph = ControlFlowGraph()
        self.current_block: BasicBlock = self.cfg.entry_block
        self.namespace = namespace or Namespace()

    def emit(self, stmt) -> None:
        """Add a statement to the current block, assigning it a unique stmt id."""
        self.current_block.add_statement(self.cfg.number_statement(stmt))

    def build(self, node: AssemblyNode) -> ControlFlowGraph:
        return self(node)

    def __call__(
        self,
        node: AssemblyNode,
        break_block: BasicBlock | None = None,
        return_block: BasicBlock | None = None,
    ) -> ControlFlowGraph:
        match node:
            case (
                Unpack()
                | Repack()
                | Resize()
                | SetAttr()
                | Print()
                | Store()
                | Assign()
                | Assert()
            ):
                self.emit(node)
            case Block(bodies):
                for body in bodies:
                    self(body, break_block, return_block)
            case If(cond, body):
                self(IfElse(cond, body, Block()), break_block, return_block)
            case IfElse(cond, body, else_body):
                before_block = self.current_block

                if_block = self.cfg.new_block()
                else_block = self.cfg.new_block()
                after_block = self.cfg.new_block()

                before_block.add_successor(if_block)
                before_block.add_successor(else_block)

                self.current_block = if_block
                self.emit(Assert(cond))
                self(body, break_block, return_block)
                self.current_block.add_successor(after_block)

                self.current_block = else_block
                self.emit(
                    Assert(
                        Call(
                            Literal(operator.not_),
                            (cond,),
                        )
                    )
                )
                self(else_body, break_block, return_block)
                self.current_block.add_successor(after_block)

                self.current_block = after_block
            case WhileLoop(cond, body):
                before_block = self.current_block

                body_block = self.cfg.new_block()
                after_block = self.cfg.new_block()

                before_block.add_successor(body_block)
                before_block.add_successor(after_block)

                self.current_block = body_block
                self.emit(Assert(cond))
                self(body, after_block, return_block)

                self.current_block.add_successor(body_block)
                self.current_block.add_successor(after_block)
                self.current_block = after_block
                self.emit(
                    Assert(
                        Call(
                            Literal(operator.not_),
                            (cond,),
                        )
                    )
                )
            case ForLoop(var, start, end, body):
                before_block = self.current_block

                fic_var_name = self.namespace.freshen("j")
                fic_var = Variable(fic_var_name, np.int64)
                self.emit(Assign(fic_var, start))

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

                self(WhileLoop(loop_condition, loop_body), break_block, return_block)
            case BufferLoop(buf, var, body):
                before_block = self.current_block

                fic_var_name = self.namespace.freshen("j")
                fic_var = Variable(fic_var_name, np.int64)
                self.emit(Assign(fic_var, Literal(np.int64(0))))

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

                self(WhileLoop(loop_condition, loop_body), break_block, return_block)
            case Return(value):
                self.emit(Return(value))
                assert return_block
                self.current_block.add_successor(return_block)
                unreachable_block = self.cfg.new_block()
                self.current_block = unreachable_block
            case Break():
                self.emit(Break())
                assert break_block
                self.current_block.add_successor(break_block)
                unreachable_block = self.cfg.new_block()
                self.current_block = unreachable_block
            case Function(_, args, body):
                for arg in args:
                    if not isinstance(arg, Variable):
                        raise NotImplementedError(f"Unrecognized argument type: {arg}")

                    # Ensure arguments appear as assigned/defined for dataflow.
                    self.emit(Assign(arg, arg))

                self(body, break_block, return_block)
            case Module(funcs):
                for func in funcs:
                    if not isinstance(func, Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )

                    if not isinstance(func.name, Variable):
                        raise NotImplementedError(
                            f"Unrecognized function name type: {type(func.name)}"
                        )
                    func_name = func.name.name

                    # set block names to the function name
                    self.cfg.block_name = func_name

                    # create entry/exit block for the function
                    func_entry_block = self.cfg.new_block()
                    func_exit_block = self.cfg.new_block()

                    # connect CFG entry block to the function's entry block
                    self.cfg.entry_block.add_successor(func_entry_block)

                    # dive into the body of the function
                    self.current_block = func_entry_block
                    self(func, break_block, func_exit_block)

                    # connect last block in the function to the
                    # exit block of the function
                    self.current_block.add_successor(func_exit_block)

                    # connect function exit block to the exit block of the CFG
                    func_exit_block.add_successor(self.cfg.exit_block)
            case node:
                raise NotImplementedError(node)

        return self.cfg
