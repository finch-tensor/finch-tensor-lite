from abc import ABC, abstractmethod


class BasicBlock:
    """A basic block of FinchAssembly Control Flow Graph."""

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
    """Control-Flow Graph (CFG) for a single FinchAssembly function."""

    def __init__(self, func_name: str):
        self.block_counter = 0
        self.name = func_name
        self.blocks: dict[str, BasicBlock] = {}

        self.entry_block = self.new_block()
        self.exit_block = self.new_block()

    def new_block(self) -> BasicBlock:
        bid = f"{self.name}_{self.block_counter}"
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


class CFGPrinterContext:
    def print(self, cfgs: dict[str, ControlFlowGraph]) -> str:
        """Print multiple CFGs in LLVM style."""
        cfg_sections = []

        for cfg in cfgs.values():
            # CFG header
            header = (
                f"{cfg.name}: #entry={cfg.entry_block.id}, #exit={cfg.exit_block.id}"
            )

            # Indent all CFG lines
            cfg_str = str(cfg)
            indented_lines = [
                f"    {line}" if line.strip() else "" for line in cfg_str.split("\n")
            ]

            # Combine header and indented content
            cfg_sections.append(header + "\n" + "\n".join(indented_lines))

        return "\n\n".join(cfg_sections)


class DataFlowAnalysis(ABC):
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg: ControlFlowGraph = cfg
        # TODO: decide on types for input/output states
        self.input_states: dict[str, dict] = {
            block.id: {} for block in cfg.blocks.values()
        }
        self.output_states: dict[str, dict] = {
            block.id: {} for block in cfg.blocks.values()
        }

    @abstractmethod
    def bottom_element(self):
        """
        Bottom element for the lattice.
        This should be implemented by subclasses.
        """
        ...

    @abstractmethod
    def top_element(self):
        """
        Top element for the lattice.
        This should be implemented by subclasses.
        """
        ...

    @abstractmethod
    def transfer(self, stmts, state: dict) -> list:
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
        return "forward"

    def analyze(self):
        """
        Perform the data flow analysis on the control flow graph.
        This method initializes the work list and processes each block.
        """
        if self.direction() == "forward":
            work_list = [self.cfg.entry_block]
            while work_list:
                block = work_list.pop(0)

                if block == self.cfg.entry_block:
                    input_state = self.input_states[block.id]
                else:
                    input_state = self.bottom_element()
                    for pred in block.predecessors:
                        input_state = self.join(
                            input_state, self.output_states.get(pred.id, {})
                        )
                    self.input_states[block.id] = input_state

                output_state = self.transfer(block.statements, input_state)

                if output_state != self.output_states.get(block.id, {}):
                    self.output_states[block.id] = output_state

                    for successor in block.successors:
                        if successor not in work_list:
                            work_list.append(successor)
        else:
            work_list = [self.cfg.exit_block]
            while work_list:
                block = work_list.pop(0)

                if block == self.cfg.exit_block:
                    input_state = self.input_states[block.id]
                else:
                    input_state = self.bottom_element()
                    for succ in block.successors:
                        input_state = self.join(
                            input_state, self.output_states.get(succ.id, {})
                        )
                    self.input_states[block.id] = input_state

                output_state = self.transfer(block.statements, input_state)

                if output_state != self.output_states.get(block.id, {}):
                    self.output_states[block.id] = output_state

                    for predecessor in block.predecessors:
                        if predecessor not in work_list:
                            work_list.append(predecessor)
