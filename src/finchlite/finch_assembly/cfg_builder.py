import operator
from dataclasses import dataclass

import numpy as np

from ..symbolic import (
    BasicBlock,
    ControlFlowGraph,
    Namespace,
    PostWalk,
    Rewrite,
)
from .nodes import (
    AssemblyNode,
    AssemblyStatement,
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


@dataclass(eq=True, frozen=True)
class NumberedStatement(AssemblyStatement):
    """
    Wrapper for AssemblyStatement that assigns a unique id to each statement
    for easier tracking in the CFG.

    Attributes:
        stmt: The original AssemblyStatement being wrapped.
        sid: A unique integer identifier for the statement.
    """

    stmt: AssemblyStatement
    sid: int

    def __str__(self) -> str:
        return str(self.stmt)


def assembly_build_cfg(
    node: AssemblyNode, namespace: Namespace | None = None
) -> ControlFlowGraph:
    """
    Build control-flow graph for a FinchAssembly node.
    Args:
        node: Root FinchAssembly node to build CFG for.
        namespace: Optional Namespace for variable name management.
    Returns:
        ControlFlowGraph: The constructed control-flow graph.
    """

    namespace = namespace or Namespace(node)
    ctx = AssemblyCFGBuilder(namespace=namespace)
    return ctx.build(node)


def assembly_desugar(root: AssemblyNode, namespace: Namespace) -> AssemblyNode:
    """
    Lower surface syntax to a core AST shape before CFG construction.

    - `If(cond, body)` -> `IfElse(cond, body, Block())`
    - `IfElse` branch bodies get leading `Assert(cond)`/`Assert(not cond)`
    - `ForLoop`/`BufferLoop` -> explicit `Assign`+`WhileLoop` with increment
    - `WhileLoop(cond, body)` gets `Assert(cond)` prepended to its body
    - `Block(..., WhileLoop(cond, ...), ...)` gets `Assert(not cond)` inserted
      immediately after each `WhileLoop` statement
    """

    def _as_not_expr(cond):
        return Call(Literal(operator.not_), (cond,))

    def go(node: AssemblyNode):
        match node:
            case Module(funcs):
                return Module(tuple(go(f) for f in funcs))
            case Function(name, args, body):
                body_2 = go(body)

                # Make argument definitions explicit so they get statement ids.
                func_prologue = tuple(Assign(arg, arg) for arg in args)
                return Function(name, args, Block((*func_prologue, *body_2.bodies)))
            case Block(bodies):
                new_bodies: list[AssemblyStatement] = []
                for b in bodies:
                    b2 = go(b)
                    new_bodies.append(b2)

                return Block(tuple(new_bodies))
            case If(cond, body):
                return go(IfElse(cond, body, Block(())))
            case IfElse(cond, body, else_body):
                then_block = go(body)
                else_block = go(else_body)

                then_block = Block((Assert(cond), *then_block.bodies))
                else_block = Block((Assert(_as_not_expr(cond)), *else_block.bodies))

                return IfElse(cond, then_block, else_block)
            case WhileLoop(cond, body):
                body_block = go(body)
                body_block = Block((Assert(cond), *body_block.bodies))

                # Insert loop-exit assertion immediately after each while.
                return Block((WhileLoop(cond, body_block), Assert(_as_not_expr(cond))))
            case ForLoop(var, start, end, body):
                fic_var_name = namespace.freshen("j")
                fic_var = Variable(fic_var_name, np.int64)

                init = Assign(fic_var, start)
                cond = Call(Literal(operator.lt), (fic_var, end))

                body_block = go(body)

                inc = Assign(
                    fic_var,
                    Call(
                        Literal(operator.add),
                        (fic_var, Literal(np.int64(1))),
                    ),
                )

                loop_body = Block((Assign(var, fic_var), *body_block.bodies, inc))
                return go(Block((init, WhileLoop(cond, loop_body))))
            case BufferLoop(buf, var, body):
                fic_var_name = namespace.freshen("j")
                fic_var = Variable(fic_var_name, np.int64)

                init = Assign(fic_var, Literal(np.int64(0)))
                cond = Call(Literal(operator.lt), (fic_var, Length(buf)))

                body_block = go(body)

                inc = Assign(
                    fic_var,
                    Call(
                        Literal(operator.add),
                        (fic_var, Literal(np.int64(1))),
                    ),
                )

                loop_body = Block(
                    (
                        Assign(var, Load(buf, fic_var)),
                        *body_block.bodies,
                        inc,
                    )
                )
                return go(Block((init, WhileLoop(cond, loop_body))))
            case node:
                return node

    return go(root)


def assembly_number_statements(root: AssemblyNode) -> AssemblyNode:
    """Wrap each statement with a NumberedStatement containing a unique id."""
    sid = 0

    def rw(x: AssemblyNode) -> AssemblyNode | None:
        nonlocal sid
        if isinstance(
            x,
            (
                Unpack,
                Repack,
                Resize,
                SetAttr,
                Print,
                Store,
                Assign,
                Assert,
                Return,
                Break,
            ),
        ):
            s = NumberedStatement(x, sid)
            sid += 1
            return s
        return None

    return Rewrite(PostWalk(rw))(root)


def assembly_unwrap_numbered_statements(node: AssemblyNode) -> AssemblyNode:
    """Remove NumberedStatement wrappers from statements."""

    def rw(x: AssemblyNode):
        match x:
            case NumberedStatement(stmt, _):
                return stmt

        return None

    return Rewrite(PostWalk(rw))(node)


def assembly_resugar(node: AssemblyNode) -> AssemblyNode:
    """
    Resugar core AST shapes back to surface syntax after CFG construction.
        - `IfElse(cond, body, Block())` -> `If(cond, body)`
        - `Assert(cond)` removed
        - Prologue in `Function` bodies of the form `Assign(arg, arg)` removed
    """

    # Helper to drop function prologue of the form `Assign(arg, arg)`.
    def _drop_function_prologue(
        args: tuple[Variable, ...], body: AssemblyStatement
    ) -> AssemblyStatement:
        if not isinstance(body, Block):
            return body

        bodies = list(body.bodies)
        idx = 0
        for arg in args:
            if idx >= len(bodies):
                break
            match bodies[idx]:
                case Assign(lhs, rhs) if lhs == arg and rhs == arg:
                    idx += 1
                    continue
            break

        if idx:
            return Block(tuple(bodies[idx:]))
        return body

    def rw(x: AssemblyNode):
        match x:
            case Function(name, args, body):
                # Drop function prologue of the form `Assign(arg, arg)`
                return Function(name, args, _drop_function_prologue(args, body))
            case Block(bodies):
                # Drop `Assert` statements.
                new_bodies = [body for body in bodies if not isinstance(body, Assert)]
                return Block(tuple(new_bodies))
            case IfElse(cond, then_body, else_body):
                # Resugar `IfElse(cond, body, Block())` to `If(cond, body)`.
                if isinstance(else_body, Block) and len(else_body.bodies) == 0:
                    return If(cond, then_body)

        return None

    return Rewrite(PostWalk(rw))(node)


class AssemblyCFGBuilder:
    """Incrementally builds control-flow graph for Finch Assembly IR."""

    def __init__(self, namespace: Namespace | None = None):
        self.cfg: ControlFlowGraph = ControlFlowGraph()
        self.current_block: BasicBlock = self.cfg.entry_block
        self.namespace = namespace or Namespace()

    def emit(self, stmt) -> None:
        """Add a statement to the current block."""
        self.current_block.add_statement(stmt)

    def build(self, node: AssemblyNode) -> ControlFlowGraph:
        return self(node)

    def __call__(
        self,
        node: AssemblyNode,
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
            case If(cond, body):
                self(IfElse(cond, body, Block()), break_block, return_block)
            case IfElse(cond, body, else_body):
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
            case WhileLoop(cond, body):
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
            case Function(_, _, body):
                # Function argument definitions are inserted by `assembly_desugar`.
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
                raise NotImplementedError(node, "AssemblyCFGBuilder")

        return self.cfg
