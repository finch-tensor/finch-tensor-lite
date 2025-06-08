from abc import ABC
from typing import List, Optional

class BasicBlock:
    def __init__(self, insts, inputs, outputs, name=None):
        self.name = name
        self.inst = insts
        self.inputs = inputs
        self.outputs = outputs
        self.successors = []
        self.predecessors = []
    
    def add_predecessor(self, block):
        self.predecessors.append(block)
        block.successors.append(self)

    def add_successor(self, block):
        self.successors.append(block)
        block.predecessors.append(self)

    def __repr__(self):
        return f"BasicBlock({self.name})"

class ControlFlowGraph:
    def __init__(self):
        self.blocks = []
        self.entry_blocks = []
        self.exit_blocks = []

    def add_block(self, block: BasicBlock):
        self.blocks.append(block)
    
    def block(self, insts, name=None):
        block = BasicBlock(insts, name)
        self.add_block(block)
        return block

    def __repr__(self):
        return f"ControlFlowGraph(entry_blocks={self.entry_blocks}, exit_blocks={self.exit_blocks}, blocks={self.blocks})"

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
    def join(self, state_1:Dict, state_2:Dict) -> Dict:
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