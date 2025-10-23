import operator

import numpy as np

from ..symbolic import (
    BasicBlock,
    ControlFlowGraph,
    DataFlowAnalysis,
    Namespace,
    PostWalk,
    Rewrite,
)
from ..util import qual_str
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
    TaggedVariable,
    Unpack,
    Variable,
    WhileLoop,
)


def assembly_copy_propagation(node: AssemblyNode):
    ctx = AssemblyCopyPropagation(assembly_build_cfg(node))
    ctx.analyze()
    return ctx


def assembly_build_cfg(node: AssemblyNode):
    ctx = AssemblyCFGBuilder(namespace=Namespace(node))

    # build cfg based on the numbered AST and return it
    return ctx.build(assembly_number_uses(node))


def assembly_number_uses(root: AssemblyNode) -> AssemblyNode:
    """
    Number every Variable occurrence in a post-order traversal.
    """
    counters: dict[str, int] = {}

    def rule(node):
        match node:
            case Variable(name, _) as var:
                idx = counters.get(name, 0)
                counters[name] = idx + 1
                return TaggedVariable(var, idx)

    return Rewrite(PostWalk(rule))(root)


class AssemblyCFGBuilder:
    """Incrementally builds control-flow graph for Finch Assembly IR."""

    def __init__(self, namespace: Namespace | None = None):
        self.cfg: ControlFlowGraph = ControlFlowGraph()
        self.current_block: BasicBlock = self.cfg.entry_block
        self.namespace = namespace or Namespace()

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
                self.current_block.add_statement(node)
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
                self.current_block.add_statement(Assert(cond))
                self(body, break_block, return_block)
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
                self.current_block.add_statement(Assert(cond))
                self(body, after_block, return_block)

                self.current_block.add_successor(before_block)
                self.current_block = after_block
                self.current_block.add_statement(
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
                fic_var = TaggedVariable(Variable(fic_var_name, np.int64), 0)
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

                self(WhileLoop(loop_condition, loop_body), break_block, return_block)
            case BufferLoop(buf, var, body):
                before_block = self.current_block

                fic_var_name = self.namespace.freshen("j")
                fic_var = TaggedVariable(Variable(fic_var_name, np.int64), 0)
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

                self(WhileLoop(loop_condition, loop_body), break_block, return_block)
            case Return(value):
                self.current_block.add_statement(Return(value))
                assert return_block
                self.current_block.add_successor(return_block)
                unreachable_block = self.cfg.new_block()
                self.current_block = unreachable_block
            case Break():
                self.current_block.add_statement(Break())
                assert break_block
                self.current_block.add_successor(break_block)
                unreachable_block = self.cfg.new_block()
                self.current_block = unreachable_block
            case Function(_, args, body):
                for arg in args:
                    match arg:
                        case TaggedVariable():
                            self.current_block.add_statement(Assign(arg, arg))
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )

                self(body, break_block, return_block)
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


class AssemblyCopyPropagation(DataFlowAnalysis):
    def direction(self) -> str:
        """
        Copy propagation is a forward analysis.
        """
        return "forward"

    def transfer(self, stmts, state: dict) -> dict:
        new_state = state.copy()

        for stmt in stmts:
            match stmt:
                case Assign(lhs, rhs):
                    # Get the variable name being assigned to
                    var_name = self._get_variable_name(lhs)

                    if var_name is not None:
                        new_state[var_name] = rhs

                        # invalidate any copies that point
                        # to the variable being assigned
                        to_remove = []
                        for var, val in new_state.items():
                            if val == lhs:
                                to_remove.append(var)

                        for var in to_remove:
                            new_state.pop(var)

        return new_state

    def join(self, state_1: dict, state_2: dict) -> dict:
        result = {}

        # only keep copy relationships that exist in both states with the same value
        for var_name in state_1:
            if var_name in state_2 and self._values_equal(
                state_1[var_name], state_2[var_name]
            ):
                result[var_name] = state_1[var_name]

        return result

    def _get_variable_name(self, var) -> str | None:
        match var:
            case Variable(name, _):
                return name
            case TaggedVariable(Variable(name, _), _):
                return name
            case _:
                return None

    def _values_equal(self, val1, val2) -> bool:
        """
        Check if two values are equal.
        """
        if isinstance(val1, (Variable, TaggedVariable)) and isinstance(
            val2, (Variable, TaggedVariable)
        ):
            return val1 == val2

        if isinstance(val1, Literal) and isinstance(val2, Literal):
            return val1.val == val2.val

        return False

    def __str__(self) -> str:
        """
        Print the CFG with dataflow annotations embedded in statements.
        Annotate variables in expressions with their known values.
        """
        lines = []
        blocks = list(self.cfg.blocks.values())

        for block in blocks:
            # block header
            if block.successors:
                succ_names = [succ.id for succ in block.successors]
                succ_str = f"#succs=[{', '.join(succ_names)}]"
            else:
                succ_str = "#succs=[]"

            lines.append(f"{block.id}: {succ_str}")

            # get the input state for this block
            input_state = self.input_states.get(block.id, {})

            # annotate each statement with analysis output
            for stmt in block.statements:
                annotated_stmt = self._annotate_statement(stmt, input_state)
                lines.append(f"    {annotated_stmt}")

            lines.append("")

        return "\n".join(lines)

    def _annotate_statement(self, stmt, state: dict) -> str:
        """Annotate a statement with dataflow information."""

        match stmt:
            case Assign(lhs, rhs):
                rhs_str = self._annotate_expression(rhs, state)
                return f"{str(lhs)} = {rhs_str}"
            case Assert(exp):
                exp_str = self._annotate_expression(exp, state)
                return f"assert({exp_str})"
            case _:
                return str(stmt)

    def _annotate_expression(self, expr, state: dict) -> str:
        """Recursively annotate an expression with dataflow information."""

        match expr:
            case Literal(value):
                return qual_str(value)
            case TaggedVariable(Variable(name, _), id):
                var_str = f"{name}_{id}"
                if name in state:
                    return f"{var_str}: {str(state[name])}"
                return var_str
            case Variable(name, _):
                if name in state:
                    return f"{name}: {str(state[name])}"
                return str(name)
            case Call(Literal(_) as lit, args):
                # annotate each argument
                annotated_args = [self._annotate_expression(arg, state) for arg in args]
                return f"{qual_str(lit.val)}({', '.join(annotated_args)})"
            case _:
                return str(expr)
