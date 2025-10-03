from abc import ABC, abstractmethod

class BasicBlock:
    """A basic block of FinchAssembly's Control Flow Graph."""

    def __init__(self, id: str) -> None:
        self.id = id
        self.statements: list = []
        self.successors: list[BasicBlock] = []
        self.predecessors: list[BasicBlock] = []

    def add_statement(self, statement) -> None:
        self.statements.append(statement)

    def add_successor(self, successor: "BasicBlock") -> None:
        if successor not in self.successors:
            self.successors.append(successor)

        if self not in successor.predecessors:
            successor.predecessors.append(self)

    def __str__(self) -> str:
        """String representation of BasicBlock in LLVM style."""
        lines = []

        if self.successors:
            succ_names = [succ.id for succ in self.successors]
            succ_str = f" #succs=[{', '.join(succ_names)}]"
        else:
            succ_str = " #succs=[]"

        # Block header
        lines.append(f"{self.id}:{succ_str}")

        # Block statements
        lines.extend(f"    {stmt}" for stmt in self.statements)

        return "\n".join(lines)

class ControlFlowGraph:
    """Control-Flow Graph (CFG) for FinchAssembly."""

    def __init__(self):
        self.block_counter = 0
        self.block_name = ""
        self.blocks: dict[str, BasicBlock] = {}

        self.entry_block = self.new_block_custom("ENTRY")
        self.exit_block = self.new_block_custom("EXIT")

    def new_block(self) -> BasicBlock:
        bid = f"{self.block_name}_{self.block_counter}"
        self.block_counter += 1
        block = BasicBlock(bid)
        self.blocks[bid] = block
        return block

    def new_block_custom(self, name: str) -> BasicBlock:
        bid = f"{name}_{self.block_counter}"
        self.block_counter += 1
        block = BasicBlock(bid)
        self.blocks[bid] = block
        return block

    def __str__(self) -> str:
        """Print the CFG in LLVM style format."""
        blocks = list(self.blocks.values())

        # Use list comprehension with join for better performance
        block_strings = [str(block) for block in blocks]
        return "\n\n".join(block_strings)


class DataFlowAnalysis(ABC):
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg: ControlFlowGraph = cfg
        self.input_states: dict[str, dict] = {
            block.id: {} for block in cfg.blocks.values()
        }
        self.output_states: dict[str, dict] = {
            block.id: {} for block in cfg.blocks.values()
        }

    @abstractmethod
    def transfer(self, stmts, state: dict) -> dict:
        """
        Transfer function for the data flow analysis.
        This should be implemented by subclasses.
        """
        ...

    @abstractmethod
    def join(self, state_1: dict, state_2: dict) -> dict:
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
        ...

    def analyze(self):
        """
        Perform the data flow analysis on the control flow graph.
        This method initializes the work list and processes each block.
        """
        if self.direction() == "forward":
            work_list = [self.cfg.entry_block]
            while work_list:
                block = work_list.pop(0)

                # get current input state based on the predecessors output states
                input_state = {}
                for pred in block.predecessors:
                    input_state = self.join(
                        input_state, self.output_states.get(pred.id, {})
                    )
                self.input_states[block.id] = input_state

                # perform trasnfer based on the statements in the current basic block
                output_state = self.transfer(block.statements, input_state)

                # check if output_state changed
                if output_state != self.output_states.get(block.id, {}):
                    self.output_states[block.id] = output_state

                    for successor in block.successors:
                        if successor not in work_list:
                            work_list.append(successor)
        else:
            work_list = [self.cfg.exit_block]
            while work_list:
                block = work_list.pop(0)

                # get current input state based on the predecessors output states
                input_state = {}
                for succ in block.successors:
                    input_state = self.join(
                        input_state, self.output_states.get(succ.id, {})
                    )
                self.input_states[block.id] = input_state

                # perform trasnfer based on the statements in the current basic block
                output_state = self.transfer(block.statements, input_state)

                # check if output state changed
                if output_state != self.output_states.get(block.id, {}):
                    self.output_states[block.id] = output_state

                    for predecessor in block.predecessors:
                        if predecessor not in work_list:
                            work_list.append(predecessor)
