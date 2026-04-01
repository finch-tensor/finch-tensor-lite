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
    FusedTree,
    If,
    Module,
    Return,
    While,
)


@dataclass(eq=True, frozen=True)
class NumberedStatement(FusedTree, FusedStatement):
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

    @property
    def children(self):
        return (self.stmt, self.sid)


def number_statements(node: FusedNode, sid: int = 0) -> tuple[FusedNode, int]:
    """
    Recursively traverse the AST and wrap each statement in a NumberedStatement with a unique id.

    Args:
        node: The root node of the AST to number.
        sid: The starting statement id for numbering. Defaults to 0.

    Returns:
        A tuple containing the new AST with numbered statements and the next available statement id.
    """

    def go(node: FusedNode) -> FusedNode:
        match node:
            case Module(funcs):
                return Module(tuple(go(f) for f in funcs))
            case Function(name, args, body):
                return Function(name, args, go(body))
            case Block(bodies):
                new_bodies: list[FusedStatement] = []
                for b in bodies:
                    b2 = go(b)
                    if isinstance(b2, Block):
                        new_bodies.extend(b2.body)
                    else:
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
                    nonlocal sid
                    s = NumberedStatement(node, sid)
                    sid += 1
                    return s
                return node

    return go(node), sid


def fused_desugar(node: FusedNode) -> tuple[FusedNode, int]:
    """
    Lower surface syntax to a core AST shape before CFG construction.

    - make function parameters explicit in the function body so they get statement ids
         and can be referenced in the CFG (e.g. desugar `def f(x): return x + 1` to
        `def f(x): x_1 = x; return x_1 + 1` so that the parameter `x` gets a statement
        id and can be referenced in the CFG)
    """

    def go(node: FusedNode) -> FusedNode:
        """Recursively desugar the AST."""
        match node:
            case Module(funcs):
                return Module(tuple(go(f) for f in funcs))
            case Function(name, args, body):
                body_2 = go(body)

                # Make argument definitions explicit so they get statement ids.
                func_prologue = tuple((Assign(arg, arg)) for arg in args)
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
                return node

    return go(node)


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
            case Assign():
                self.emit(node)
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
            case If(cond, body, else_body):
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
                self(body, break_block, return_block)
                self.current_block.add_successor(after_block)

                # fill in the else block
                self.current_block = else_block
                self(else_body, break_block, return_block)
                self.current_block.add_successor(after_block)

                # continue building after the if-else
                self.current_block = after_block
            case While(cond, body):
                before_block = self.current_block
                self.emit(cond)

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
            case For(iter_var, iter, body):
                before_block = self.current_block
                body_block = self.cfg.new_block()
                after_block = self.cfg.new_block()

                # connect before block to the loop body and the after block
                before_block.add_successor(body_block)
                before_block.add_successor(after_block)

                # fill in the loop body
                self.current_block = body_block
                self.emit(iter)
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


def fused_build_cfg(node: FusedNode) -> ControlFlowGraph:
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
    ctx = FusedCFGBuilder()
    return ctx.build(node)
