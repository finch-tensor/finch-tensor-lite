from dataclasses import dataclass

from ..symbolic.dataflow import BasicBlock, ControlFlowGraph
from .nodes import (
    Assign,
    Block,
    Break,
    For,
    Function,
    FusedNode,
    FusedStatement,
    If,
    Module,
    Return,
    While,
)


@dataclass(eq=True, frozen=True)
class NumberedStatement(FusedStatement):
    """
    Wrapper for AssemblyStatement that assigns a unique id to each statement
    for easier tracking in the CFG.

    Attributes:
        stmt: The original FusedStatement being wrapped.
        sid: A unique integer identifier for the statement.
    """

    stmt: FusedStatement
    sid: int

    def __str__(self) -> str:
        return str(self.stmt)


def fused_desugar(node: FusedNode, sid: int = 0) -> tuple[FusedNode, int]:
    """
    Lower surface syntax to a core AST shape before CFG construction.

    - number the statements in the AST with unique ids for easier tracking in the CFG
    - make function parameters explicit in the function body so they get statement ids
         and can be referenced in the CFG (e.g. desugar `def f(x): return x + 1` to
        `def f(x): x_1 = x; return x_1 + 1` so that the parameter `x` gets a statement
        id and can be referenced in the CFG)
    """

    def _number_stmt(stmt: FusedStatement) -> NumberedStatement:
        """Helper to wrap a statement in a NumberedStatement with a unique id."""
        nonlocal sid
        s = NumberedStatement(stmt, sid)
        sid += 1
        return s

    def go(node: FusedNode) -> FusedNode:
        """Recursively desugar the AST."""
        match node:
            case Module(funcs):
                return Module(tuple(go(f) for f in funcs))
            case Function(name, args, body):
                body_2 = go(body)

                # Make argument definitions explicit so they get statement ids.
                func_prologue = tuple(_number_stmt(Assign(arg, arg)) for arg in args)

                return Function(name, args, Block((*func_prologue, *body_2.body)))
            case Block(bodies):
                new_bodies: list[FusedStatement] = []
                for b in bodies:
                    b2 = go(b)
                    new_bodies.append(b2)
                return Block(tuple(new_bodies))
            case If(cond, body, else_body):
                return If(cond, go(body), go(else_body))
            case While(cond, body):
                return While(cond, go(body))
            case For(target, iter, body):
                return For(target, iter, go(body))
            case node:
                if isinstance(node, (Assign, Return, Break)):
                    return _number_stmt(node)
                return node

    return go(node), sid


class FusedCFGBuilder:
    """Incrementally builds control-flow graph for Finch Fused IR."""

    def __init__(self):
        self.cfg: ControlFlowGraph = ControlFlowGraph()
        self.current_block: BasicBlock = self.cfg.entry_block

    def emit(self, stmt) -> None:
        """Add a statement to the current block."""
        self.current_block.add_statement(stmt)

    def build(self, node: FusedNode) -> ControlFlowGraph:
        return self(node)

    def __call__(
        self,
        node: FusedNode,
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
                    case Break():
                        self.emit(node)
                        assert break_block
                        self.current_block.add_successor(break_block)
                        unreachable_block = self.cfg.new_block()
                        self.current_block = unreachable_block
                    case _:
                        self.emit(node)
            case Block(bodies):
                for body in bodies:
                    self(body, break_block, return_block)
            case If(_, body, else_body):
                before_block = self.current_block

                # create blocks for if, else, and after
                if_block = self.cfg.new_block()
                else_block = self.cfg.new_block()
                after_block = self.cfg.new_block()

                # connect before block to if and else blocks
                before_block.add_successor(if_block)
                before_block.add_successor(else_block)

                # fill in the if block
                self.current_block = if_block
                self(body, break_block, return_block)
                self.current_block.add_successor(after_block)

                # fill in the else block
                self.current_block = else_block
                self(else_body, break_block, return_block)
                self.current_block.add_successor(after_block)

                # continue building after the if-else
                self.current_block = after_block
            case While(_, body):
                before_block = self.current_block

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
            case For(_, _, body):
                before_block = self.current_block
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

            case Function(func_name, _, body):
                # set block names to the function name
                self.cfg.block_name = func_name

                # create entry/exit block for the function
                func_entry_block = self.cfg.new_block()
                func_exit_block = self.cfg.new_block()

                # connect CFG entry block to the function's entry block
                self.cfg.entry_block.add_successor(func_entry_block)

                # dive into the body of the function
                self.current_block = func_entry_block
                self(body, break_block, func_exit_block)

                # connect last block in the function to the
                # exit block of the function
                self.current_block.add_successor(func_exit_block)

                # connect function exit block to the exit block of the CFG
                func_exit_block.add_successor(self.cfg.exit_block)

            case node:
                raise NotImplementedError(node, "FusedCFGBuilder")

        return self.cfg


def fused_build_cfg(node: FusedNode, sid: int) -> ControlFlowGraph:
    """
    Build control-flow graph for a FinchAssembly node.
    Args:
        node: Root FinchAssembly node to build CFG for.
        sid: Starting statement id for numbering additional
            statements during CFG desugaring.
        namespace: Namespace for variable name management.
    Returns:
        ControlFlowGraph: The constructed control-flow graph.
    """

    # desugar the input name and number additional statements for CFG construction
    desugared_node, sid = fused_desugar(node, sid)

    ctx = FusedCFGBuilder()
    return ctx.build(desugared_node)
