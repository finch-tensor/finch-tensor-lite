from __future__ import annotations

from dataclasses import dataclass

from ..symbolic import BasicBlock, ControlFlowGraph, PostWalk, Rewrite
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
        return f"[{self.sid}] {str(self.stmt)}"

    @property
    def children(self):
        return (self.stmt, self.sid)


def notation_build_cfg(node: NotationNode, sid: int) -> ControlFlowGraph:
    """
    Build control-flow graph for a FinchNotation node.
    Args:
        node: Root FinchNotation node to build CFG for.
        sid: Starting statement id for numbering additional
            statements during CFG desugaring.
    Returns:
        ControlFlowGraph: The constructed control-flow graph.
    """

    # desugar the input and number additional statements for CFG construction
    desugared_node = notation_desugar(root=node, sid=sid)

    ctx = NotationCFGBuilder()
    return ctx.build(desugared_node)


def notation_desugar(root: NotationNode, sid: int) -> NotationNode:
    """
    Lower surface syntax to a core AST shape before CFG construction.

    - `If(cond, body)` -> `IfElse(cond, body, Block())`
    - function parameters get explicit numbered `Assign(arg, arg)` statements
      at function entry so they can be referenced in the CFG
    """

    def _number_stmt(stmt: NotationStatement) -> NumberedStatement:
        """Helper to wrap a statement in a NumberedStatement with a unique id."""
        nonlocal sid
        s = NumberedStatement(stmt, sid)
        sid += 1
        return s

    def go(node: NotationNode):
        """Recursively desugar the AST."""
        match node:
            case Module(funcs):
                return Module(tuple(go(f) for f in funcs))
            case Function(name, args, body):
                body_2 = go(body)

                # Make argument definitions explicit so they get statement ids.
                func_prologue = tuple(_number_stmt(Assign(arg, arg)) for arg in args)

                match body_2:
                    case Block(bodies):
                        return Function(name, args, Block((*func_prologue, *bodies)))
                    case body_stmt:
                        return Function(
                            name,
                            args,
                            Block((*func_prologue, body_stmt)),
                        )
            case Block(bodies):
                new_bodies: list[NotationStatement] = []
                for b in bodies:
                    b2 = go(b)
                    new_bodies.append(b2)

                return Block(tuple(new_bodies))
            case If(cond, body):
                return go(IfElse(cond, body, Block(())))
            case IfElse(cond, body, else_body):
                return IfElse(cond, go(body), go(else_body))
            case Loop(idx, ext, body):
                return Loop(idx, ext, go(body))
            case node:
                return node

    return go(root)


def notation_dataflow_preprocess(node: NotationNode) -> tuple[NotationNode, int]:
    """
    Preprocess a FinchNotation node for dataflow analysis (number statements).
    Args:
        node: Root FinchNotation node to preprocess.
    Returns:
        NotationNode: The preprocessed FinchNotation node.
        sid: first unused statement id after preprocessing.
    """

    sid = 0

    def rw(x: NotationNode) -> NotationNode | None:
        nonlocal sid
        match x:
            case (
                Unpack()
                | Repack()
                | Declare()
                | Freeze()
                | Thaw()
                | Assign()
                | Increment()
                | Return()
            ):
                s = NumberedStatement(x, sid)
                sid += 1
                return s
            case _:
                return None

    return Rewrite(PostWalk(rw))(node), sid


def notation_dataflow_postprocess(node: NotationNode) -> NotationNode:
    """
    Postprocess a FinchNotation node after
    dataflow analysis (remove numbering).
    Args:
        node: Root FinchNotation node to postprocess.
    Returns:
        NotationNode: The postprocessed FinchNotation node.
    """

    def rw(x: NotationNode):
        match x:
            case NumberedStatement(stmt, _):
                return stmt

        return None

    return Rewrite(PostWalk(rw))(node)


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

                # create blocks for if, else, and after
                if_block = self.cfg.new_block()
                else_block = self.cfg.new_block()
                after_block = self.cfg.new_block()

                # connect before block to if and else blocks
                before_block.add_successor(if_block)
                before_block.add_successor(else_block)

                # fill in the if block
                self.current_block = if_block
                self(then_body, break_block, return_block)
                self.current_block.add_successor(after_block)

                # fill in the else block
                self.current_block = else_block
                self(else_body, break_block, return_block)
                self.current_block.add_successor(after_block)

                # continue building after the if-else
                self.current_block = after_block
            case Loop(_, ext, body):
                before_block = self.current_block
                # Record the loop extent expression (data dependency).
                self.emit(ext)

                # create blocks for the loop body and the code after the loop
                body_block = self.cfg.new_block()
                after_block = self.cfg.new_block()

                # connect before block to the loop body and the after block
                before_block.add_successor(body_block)
                before_block.add_successor(after_block)

                # fill in the loop body
                self.current_block = body_block
                self(body, after_block, return_block)

                # connect the end of loop body back to the beginning to form the loop
                self.current_block.add_successor(body_block)
                self.current_block.add_successor(after_block)
                self.current_block = after_block
            case Function(_, _, body):
                # Function argument definitions are inserted by `notation_desugar`.
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
            case other:
                raise NotImplementedError(other, "NotationCFGBuilder")

        return self.cfg
