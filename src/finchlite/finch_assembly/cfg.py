import json
import operator
from abc import ABC, abstractmethod

import numpy as np
from typing import Dict, List

from ..codegen import NumpyBuffer
from .nodes import (
    AssemblyNode,
    AssemblyPrinterContext,
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
    Unpack,
    Variable,
    WhileLoop,
)

"""
Dr. Willow Ahrens Office Hours Questions

1) What is the best representation for a 'statement'?
2) What is the first dataflow-analysis that I should do?
3) Figure out a way to number the actual AST so that I can change it later based on the output from the dataflow
4) Come up with a way to test CFG Builder in a form of unit tests and test the dataflow analysis output
5) Am I sure that I'm supposed to make a separate CFG for every function in the Module?
"""

class BasicBlock:
    def __init__(self, id):
        self.id = id
        self.statements = []
        self.successors = []
        self.predecessors = []

    def add_statement(self, statement):
        self.statements.append(statement)

    def add_successor(self, successor: "BasicBlock") -> None:
        if successor not in self.successors:
            self.successors.append(successor)

        if self not in successor.predecessors:
            successor.predecessors.append(self)

    def to_dict(self):
        """Convert BasicBlock to a dictionary for JSON serialization."""
        return {
            "id": self.id,
            "statements": [str(stmt) for stmt in self.statements],
            "successors": [str(block.id) for block in self.successors],
            "predecessors": [str(block.id) for block in self.predecessors],
        }

    def __repr__(self):
        return (
            f"BasicBlock(id={self.id}",
            f"stmts={self.statements}, succs={self.successors})",
        )


class ControlFlowGraph:
    def __init__(self, func_name: str):
        self.block_counter = 0
        self.name = func_name
        self.blocks = {}

        # initialize ENTRY and EXIT blocks
        self.entry_block = self.new_block()
        self.exit_block = self.new_block()

        self.current_block = self.new_block()
        self.entry_block.add_successor(self.current_block)

    def new_block(self):
        bid = f"{self.name}_{self.block_counter}"
        self.block_counter += 1
        block = BasicBlock(bid)
        self.blocks[bid] = block
        return block

    def to_dict(self):
        """Convert CFG to a dictionary for JSON serialization."""
        return {
            "name": self.name,
            "entry_block": self.entry_block.id,
            "exit_block": self.exit_block.id,
            "blocks": {
                block_id: block.to_dict() for block_id, block in self.blocks.items()
            },
        }


class CFGBuilder:
    """
    cfgs: {
        function_0: {
            basic_block_0: {
                statements:...,
                predecessors:...,
                successors:...,
            },
        },
        ...
    }
    """

    def __init__(self):
        self.cfgs = {}
        self.current_cfg = None

    def new_cfg(self, name: str) -> ControlFlowGraph:
        new_cfg = ControlFlowGraph(name)
        self.cfgs[name] = new_cfg
        return new_cfg

    def build(self, node: AssemblyNode):
        return self(node)

    def to_dict(self):
        """Convert all CFGs to dictionaries for JSON serialization."""
        return {cfg_name: cfg.to_dict() for cfg_name, cfg in self.cfgs.items()}

    def __call__(self, node: AssemblyNode, break_block_id: str = None):
        match node:
            case Literal(value):
                self.current_cfg.current_block.add_statement(("literal", value))
            case Unpack(lhs, rhs):
                self.current_cfg.current_block.add_statement(("unpack", lhs, rhs))
            case Repack(val):
                self.current_cfg.current_block.add_statement(("repack", val))
            case Resize(buffer, new_size):
                self.current_cfg.current_block.add_statement(
                    ("resize", buffer, new_size)
                )
            case Variable(name, type):
                self.current_cfg.current_block.add_statement(("variable", name, type))
            case GetAttr(obj, attr):
                self.current_cfg.current_block.add_statement(("getattr", obj, attr))
            case SetAttr(obj, attr, value):
                self.current_cfg.current_block.add_statement(
                    ("setattr", obj, attr, value)
                )
            case Call(Literal(_) as lit, args):
                self.current_cfg.current_block.add_statement(
                    ("function_call", lit, args)
                )
            case Load(buffer, index):
                self.current_cfg.current_block.add_statement(("load", buffer, index))
            case Store(buffer, index, value):
                self.current_cfg.current_block.add_statement(
                    ("store", buffer, index, value)
                )
            case Length(buffer):
                self.current_cfg.current_block.add_statement(("length", buffer))
            case Slot(name, type):
                self.current_cfg.current_block.add_statement(("slot", name, type))
            case Stack(obj, type):
                self.current_cfg.current_block.add_statement(("stack", obj, type))
            case Assign(Variable(name, _), val):
                self.current_cfg.current_block.add_statement(("assign", name, val))
            case Block(bodies):
                for body in bodies:
                    self(body, break_block_id)
            case If(cond, body):
                cond_block = self.current_cfg.current_block
                cond_block.add_statement(("if_cond", cond))

                if_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                cond_block.add_successor(if_block)
                cond_block.add_successor(after_block)

                self.current_cfg.current_block = if_block
                self(body, break_block_id)

                self.current_cfg.current_block.add_successor(after_block)

                self.current_cfg.current_block = after_block
            case IfElse(cond, body, else_body):
                cond_block = self.current_cfg.current_block
                cond_block.add_statement(("if_else_cond", cond))

                if_block = self.current_cfg.new_block()
                else_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                cond_block.add_successor(if_block)
                cond_block.add_successor(else_block)

                self.current_cfg.current_block = if_block
                self(body, break_block_id)
                self.current_cfg.current_block.add_successor(after_block)

                self.current_cfg.current_block = else_block
                self(else_body, break_block_id)
                self.current_cfg.current_block.add_successor(after_block)

                self.current_cfg.current_block = after_block
            case WhileLoop(cond, body):
                cond_block = self.current_cfg.current_block
                cond_block.add_statement(("while_cond", cond))

                body_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                cond_block.add_successor(body_block)
                cond_block.add_successor(after_block)

                self.current_cfg.current_block = body_block
                self(body, after_block.id)

                self.current_cfg.current_block.add_successor(cond_block)
                self.current_cfg.current_block = after_block
            case ForLoop(var, start, end, body):
                init_block = self.current_cfg.current_block
                init_block.add_statement(("for_init", var, start, end))

                cond_block = self.current_cfg.new_block()
                body_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                init_block.add_successor(cond_block)
                cond_block.add_statement(("for_init", var, start, end))

                cond_block.add_successor(body_block)
                cond_block.add_successor(after_block)

                self.current_cfg.current_block = body_block
                self(body, after_block.id)

                self.current_cfg.current_block.add_statement(("for_inc", var))
                self.current_cfg.current_block.add_successor(cond_block)

                self.current_cfg.current_block = after_block
            case BufferLoop(buf, var, body):
                init_block = self.current_cfg.current_block
                init_block.add_statement(("bufferloop_init", buf, var))

                cond_block = self.current_cfg.new_block()
                init_block.add_successor(cond_block)
                cond_block.add_statement(("bufferloop_init", var, start, end))

                body_block = self.current_cfg.new_block()
                cond_block.add_successor(body_block)

                after_block = self.current_cfg.new_block()
                cond_block.add_successor(after_block)

                self.current_cfg.current_block = body_block
                self(body_block, after_block.id)

                self.current_cfg.current_block.add_statement(("bufferloop_inc", var))
                self.current_cfg.current_block.add_successor(cond_block)

                self.current_cfg.current_block = after_block
            case Return(value):
                self.current_cfg.current_block.add_statement(("return", value))

                # when Return is met,
                # make a connection to the EXIT block of function (cfg)
                self.current_cfg.current_block.add_successor(
                    self.current_cfg.exit_block
                )

                # create a block where we going to store all unreachable statements
                unreachable_block = self.current_cfg.new_block()
                self.current_cfg.current_block = unreachable_block
            case Break():
                self.current_cfg.current_block.add_statement("break")

                # when Break is met,
                # make a connection to the AFTER block of ForLoop/WhileLoop
                self.current_cfg.current_block.add_successor(
                    break_block_id, self.current_cfg.blocks
                )

                # create a block where we going to store all unreachable statements
                unreachable_block = self.current_cfg.new_block()
                self.current_cfg.current_block = unreachable_block
            case Function(Variable(_, _), args, body):
                for arg in args:
                    match arg:
                        case Variable(name, type):
                            self.current_cfg.current_block.add_statement(
                                (("func_arg"), name, type)
                            )
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

                    self.current_cfg = self.new_cfg(func.name.name)
                    self(func)
                    self.current_cfg.current_block.add_successor(
                        self.current_cfg.exit_block
                    )
            case node:
                raise NotImplementedError(
                    f"Unhandled node type: {node.__class__.__name__}"
                )

        return self.cfgs


class DataFlowAnalysis(ABC):
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg = cfg
        self.inputs = {block: {} for block in cfg.blocks}
        self.outputs = {block: {} for block in cfg.blocks}

    @abstractmethod
    def transfer(self, insts, state: Dict) -> List:
        """
        Transfer function for the data flow analysis.
        This should be implemented by subclasses.
        """
        ...
    
    @abstractmethod
    def join(self, state_1: Dict, state_2: Dict) -> Dict:
        """
        Join function for the data flow analysis.
        This should be implemented by subclasses.
        """
        ...

    @abstractmethod
    def direction(self) -> str:
        """
        Return the direction of the data flow analysis, either "forward" or "backward".
        This should be implemented by subclasses.
        """
        return "forward"
    
    # TODO: change based on my definition of the CFG
    def analyze(self):
        """
        Perform the data flow analysis on the control flow graph.
        This method initializes the work list and processes each block.
        """
        if self.direction() == "forward":
            work_list = self.cfg.entry[:]
            while work_list:
                block = work_list.pop(0)
                input_state = self.input_states.get(block, {})
                output_state = self.transfer(block, input_state)
                if output_state != self.output_states.get(block, {}):
                    self.output_states[block] = output_state
                    for successor in block.successors:
                        if successor not in work_list:
                            work_list.append(successor)

# TODO: place tests in a separate folder (/tests)
def test1():
    printer = AssemblyPrinterContext()

    var = Variable("a", np.int64)
    root = Module(
        (
            Function(
                Variable("if_else", np.int64),
                (),
                Block(
                    (
                        Assign(var, Literal(np.int64(5))),
                        If(
                            Call(
                                Literal(operator.eq),
                                (var, Literal(np.int64(5))),
                            ),
                            Block(
                                (
                                    Assign(
                                        var,
                                        Call(
                                            Literal(operator.add),
                                            (var, Literal(np.int64(10))),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        IfElse(
                            Call(
                                Literal(operator.lt),
                                (var, Literal(np.int64(15))),
                            ),
                            Block(
                                (
                                    Assign(
                                        var,
                                        Call(
                                            Literal(operator.sub),
                                            (var, Literal(np.int64(3))),
                                        ),
                                    ),
                                )
                            ),
                            Block(
                                (
                                    Assign(
                                        var,
                                        Call(
                                            Literal(operator.mul),
                                            (var, Literal(np.int64(2))),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        Return(var),
                    )
                ),
            ),
        )
    )

    cfg_builder = CFGBuilder()
    cfg_builder.build(root)

    printer(root)
    print(printer.emit())
    print(50 * "=")

    return cfg_builder.to_dict()
def test2():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)

    # Simple dot product using numpy for expected result
    c = Variable("c", np.float64)
    i = Variable("i", np.int64)
    ab = NumpyBuffer(a)
    bb = NumpyBuffer(b)
    ab_v = Variable("a", ab.ftype)
    ab_slt = Slot("a_", ab.ftype)
    bb_v = Variable("b", bb.ftype)
    bb_slt = Slot("b_", bb.ftype)

    root = Module(
        (
            Function(
                Variable("dot_product", np.float64),
                (
                    ab_v,
                    bb_v,
                ),
                Block(
                    (
                        Assign(c, Literal(np.float64(0.0))),
                        Unpack(ab_slt, ab_v),
                        Unpack(bb_slt, bb_v),
                        ForLoop(
                            i,
                            Literal(np.int64(0)),
                            Length(ab_slt),
                            Block(
                                (
                                    Assign(
                                        c,
                                        Call(
                                            Literal(operator.add),
                                            (
                                                c,
                                                Call(
                                                    Literal(operator.mul),
                                                    (
                                                        Load(ab_slt, i),
                                                        Load(bb_slt, i),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        Repack(ab_slt),
                        Repack(bb_slt),
                        Return(c),
                    )
                ),
            ),
        )
    )

    printer = AssemblyPrinterContext()
    cfg_builder = CFGBuilder()
    cfg_builder.build(root)

    printer(root)
    print(printer.emit())
    print(50 * "=")

    return cfg_builder.to_dict()

if __name__ == "__main__":
    print(json.dumps(test1(), indent=4))
    print(50 * "=")
    print(50 * "=")
    print(json.dumps(test2(), indent=4))
