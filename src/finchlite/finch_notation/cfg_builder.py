from __future__ import annotations

from dataclasses import dataclass

from ..symbolic import BasicBlock, ControlFlowGraph
from .nodes import (
    Assign,
    Block,
    Declare,
    Freeze,
    Function,
    If,
    IfElse,
    Increment,
    Loop,
    Module,
    NotationExpression,
    NotationNode,
    NotationStatement,
    Repack,
    Return,
    Thaw,
    Unpack,
    Variable,
)


@dataclass(eq=True, frozen=True)
class NumberedStatement(NotationStatement):
    """Wrapper for NotationStatement that assigns a unique id to each statement."""

    stmt: NotationStatement
    sid: int

    def __str__(self) -> str:
        return str(self.stmt)


class NotationCFGBuilder:
    """Incrementally builds control-flow graph for Finch Notation IR."""

    def __init__(self) -> None:
        self.cfg: ControlFlowGraph = ControlFlowGraph()
        self.current_block: BasicBlock = self.cfg.entry_block

    def emit(self, stmt) -> None:
        self.current_block.add_statement(stmt)

    def build(self, node: NotationNode) -> ControlFlowGraph:
        return self(node)

    def __call__(
        self,
        node: NotationNode,
        break_block: BasicBlock | None = None,
        return_block: BasicBlock | None = None,
    ) -> ControlFlowGraph:
        match node:
            case NumberedStatement(stmt, _):
                match stmt:
                    case Return(_):
                        self.emit(node)
                        assert return_block
                        self.current_block.add_successor(return_block)
                        unreachable_block = self.cfg.new_block()
                        self.current_block = unreachable_block
                    case _:
                        self.emit(node)
            case Return(_):
                # Allow building CFGs from unnumbered ASTs.
                self.emit(node)
                assert return_block
                self.current_block.add_successor(return_block)
                unreachable_block = self.cfg.new_block()
                self.current_block = unreachable_block
            case Block(bodies):
                for body in bodies:
                    self(body, break_block, return_block)
            case (
                Assign()
                | Increment()
                | Unpack()
                | Repack()
                | Declare()
                | Freeze()
                | Thaw()
            ):
                self.emit(node)
            case If(cond, body):
                # Treat `If` as `IfElse(cond, body, Block(()))`.
                self(IfElse(cond, body, Block(())), break_block, return_block)
            case IfElse(cond, then_body, else_body):
                before_block = self.current_block
                self.emit(cond)

                if_block = self.cfg.new_block()
                else_block = self.cfg.new_block()
                after_block = self.cfg.new_block()

                before_block.add_successor(if_block)
                before_block.add_successor(else_block)

                self.current_block = if_block
                self(then_body, break_block, return_block)
                self.current_block.add_successor(after_block)

                self.current_block = else_block
                self(else_body, break_block, return_block)
                self.current_block.add_successor(after_block)

                self.current_block = after_block
            case Loop(_, ext, body):
                before_block = self.current_block
                # Record the loop extent expression (data dependency).
                self.emit(ext)

                body_block = self.cfg.new_block()
                after_block = self.cfg.new_block()

                before_block.add_successor(body_block)
                before_block.add_successor(after_block)

                self.current_block = body_block
                self(body, after_block, return_block)

                self.current_block.add_successor(body_block)
                self.current_block.add_successor(after_block)
                self.current_block = after_block
            case Function(Variable(func_name, _), _, body):
                self.cfg.block_name = func_name

                func_entry_block = self.cfg.new_block()
                func_exit_block = self.cfg.new_block()

                self.cfg.entry_block.add_successor(func_entry_block)
                self.current_block = func_entry_block
                self(body, break_block, func_exit_block)
                self.current_block.add_successor(func_exit_block)
                func_exit_block.add_successor(self.cfg.exit_block)
            case Module(funcs):
                for func in funcs:
                    if not isinstance(func, Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func, break_block, return_block)
            case expr if isinstance(expr, NotationExpression):
                # Expressions may be emitted by callers (e.g. cond/ext).
                self.emit(expr)
            case other:
                raise NotImplementedError(other, "NotationCFGBuilder")

        return self.cfg
