import operator

import numpy as np

from ..symbolic.dataflow import BasicBlock, CFGCollection, ControlFlowGraph, DataFlowAnalysis
from ..symbolic.gensym import gensym
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


class FinchAssemblyCFGBuilder:
    """Incrementally builds control-flow graphs for Finch Assembly IR."""

    def __init__(self):
        self.cfgs: CFGCollection = CFGCollection()
        self.current_block = None
        self.current_cfg: ControlFlowGraph
        self.loop_counter_id = 0

    def new_cfg(self, name: str) -> ControlFlowGraph:
        new_cfg = ControlFlowGraph(name)
        self.current_block = new_cfg.new_block()
        new_cfg.entry_block.add_successor(self.current_block)
        self.cfgs[name] = new_cfg
        return new_cfg

    def __call__(self, node: AssemblyNode, break_block: BasicBlock | None = None):
        match node:
            case (
                Unpack()
                | Repack()
                | Resize()
                | SetAttr()
                | Call()
                | Store()
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

                # create fictitious variable
                fic_var = TaggedVariable(Variable(gensym("j"), np.int64), 0)
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

                # create fictitious variable
                fic_var = TaggedVariable(Variable(gensym("j"), np.int64), 0)
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
                            self.current_block.add_statement(Assign(arg, arg))
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )

                self(body, break_block)
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
                    self(func, break_block)
                    self.current_block.add_successor(self.current_cfg.exit_block)
            case node:
                raise NotImplementedError(node)

        return self.cfgs

class FinchAssemblyCopyPropagation(DataFlowAnalysis):
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
                            
                        # invalidate any copies that point to the variable being assigned
                        to_remove = []
                        for var, val in new_state.items():
                            if self._variables_equal(val, lhs):
                                to_remove.append(var)
                        
                        for var in to_remove:
                            new_state.pop(var)
        
        return new_state

    def join(self, state_1: dict, state_2: dict) -> dict:
        result = {}
        
        # Only keep copy relationships that exist in both states with the same value
        for var_name in state_1:
            if var_name in state_2 and self._values_equal(state_1[var_name], state_2[var_name]):
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
    
    def _variables_equal(self, var1, var2) -> bool:
        name1 = self._get_variable_name(var1)
        name2 = self._get_variable_name(var2)
        return name1 is not None and name1 == name2
    
    def _values_equal(self, val1, val2) -> bool:
        """
        Check if two values are equal.
        """
        if isinstance(val1, (Variable, TaggedVariable)) and isinstance(val2, (Variable, TaggedVariable)):
            return self._variables_equal(val1, val2)
        
        if isinstance(val1, Literal) and isinstance(val2, Literal):
            return val1.val == val2.val
        
        return False