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
