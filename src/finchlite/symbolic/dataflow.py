import copy
from abc import ABC, abstractmethod


class BasicBlock:
    """Linear sequence of statements with a single entry and exit.

    Successor/predecessor lists encode the CFG edges. Statements execute in
    order; no internal branching occurs within a BasicBlock.
    """

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
        for stmt in self.statements:
            sid = getattr(stmt, "sid", None)
            prefix = f"[{sid}] " if sid is not None else ""
            lines.append(f"    {prefix}{stmt}")

        return "\n".join(lines)


class ControlFlowGraph:
    """Collection of BasicBlocks plus explicit ENTRY and EXIT nodes.

    Provides helpers to allocate uniquely named blocks and holds the global
    block mapping used by dataflow analyses.
    """

    def __init__(self) -> None:
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
    """
    Base class for performing data flow analyses over a ControlFlowGraph.

    This abstract class implements a generic work-list algorithm that can run either
    in a "forward" or "backward" direction. Subclasses must provide IR-specific
    semantics by implementing the abstract methods: transfer, join, stmt_str and
    direction.

    Key attributes
    - cfg: the ControlFlowGraph to analyze.
    - input_states: a mapping from basic block id to the lattice state that holds
        at the entry of the block.
    - output_states: a mapping from basic block id to the lattice state that holds
        at the exit of the block.
    """

    def __init__(self, cfg: ControlFlowGraph):
        self.cfg: ControlFlowGraph = cfg
        self.input_states: dict[str, dict] = {
            block.id: {} for block in cfg.blocks.values()
        }
        self.output_states: dict[str, dict] = {
            block.id: {} for block in cfg.blocks.values()
        }

    def __str__(self) -> str:
        """
        Return a string representation describing the current dataflow
        analysis state over the control-flow graph (CFG).

        The output lists each basic block in the order
        produced by self.cfg.blocks.values().
        For every block the printed form is:

            <block.id>: #succs=[<succ_id>, ...]
                <stmt_representation_for_first_stmt>
                <stmt_representation_for_second_stmt>
                ...
        """
        lines: list[str] = []
        blocks = list(self.cfg.blocks.values())

        for block in blocks:
            # get successors of the block
            if block.successors:
                succ_names = [succ.id for succ in block.successors]
                succ_str = f"#succs=[{', '.join(succ_names)}]"
            else:
                succ_str = "#succs=[]"

            lines.append(f"{block.id}: {succ_str}")

            # get the input state for this block
            input_state = self.input_states.get(block.id, {})

            if self.direction() == "forward":
                # deep copy the input_state, otherwise original state
                # in the input_states will be changed,
                # which can lead to errors in future
                state = copy.deepcopy(input_state)

                # print each statement using subclasses stmt_str method
                for stmt in block.statements:
                    sid = getattr(stmt, "sid", None)
                    prefix = f"[{sid}] " if sid is not None else ""
                    lines.append(f"    {prefix}{self.stmt_str(stmt, state)}")
                    # advance state with current statement
                    state = self.transfer([stmt], state)
            else:
                # TODO: implement "backward" direction printing
                lines.extend(
                    f"    {self.stmt_str(stmt, input_state)}"
                    for stmt in block.statements
                )

            lines.append("")

        return "\n".join(lines)

    @abstractmethod
    def stmt_str(self, stmt, state: dict) -> str:
        """
        Format a single statement given the current lattice state.
        Implement in subclasses or IR-specific abstract bases.
        """
        ...

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

    def analyze(self) -> None:
        """
        Perform the data flow analysis on the control flow graph.
        This method initializes the work list and processes each block.
        """
        work_list: list[BasicBlock] = list(self.cfg.blocks.values())
        if self.direction() == "forward":
            while work_list:
                block = work_list.pop(0)

                # get current input state based on the predecessors output states
                if not block.predecessors:
                    input_state = {}
                else:
                    pred_outputs = [
                        self.output_states.get(pred.id, {})
                        for pred in block.predecessors
                    ]
                    input_state = pred_outputs[0].copy()
                    for pred_output in pred_outputs[1:]:
                        input_state = self.join(input_state, pred_output)

                self.input_states[block.id] = input_state

                # perform transfer based on the statements in the current basic block
                output_state = self.transfer(block.statements, input_state)

                # check if output_state changed
                if output_state != self.output_states.get(block.id, {}):
                    self.output_states[block.id] = output_state

                    for successor in block.successors:
                        if successor not in work_list:
                            work_list.append(successor)
        else:
            while work_list:
                block = work_list.pop(0)

                # get current input state based on the successors output states
                if not block.successors:
                    input_state = {}
                else:
                    succ_outputs = [
                        self.output_states.get(succ.id, {}) for succ in block.successors
                    ]
                    input_state = succ_outputs[0].copy()
                    for succ_output in succ_outputs[1:]:
                        input_state = self.join(input_state, succ_output)

                self.input_states[block.id] = input_state

                # perform trasnfer based on the statements in the current basic block
                output_state = self.transfer(block.statements, input_state)

                # check if output state changed
                if output_state != self.output_states.get(block.id, {}):
                    self.output_states[block.id] = output_state

                    for predecessor in block.predecessors:
                        if predecessor not in work_list:
                            work_list.append(predecessor)
