class BasicBlock:
    """A basic block of FinchAssembly Control Flow Graph."""

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

    def __str__(self):
        import json as _json

        return _json.dumps(self.to_dict(), indent=4)


class ControlFlowGraph:
    """Control-Flow Graph (CFG) for a single FinchAssembly function."""

    def __init__(self, func_name: str):
        self.block_counter = 0
        self.name = func_name
        self.blocks: dict[str, BasicBlock] = {}

        # initialize ENTRY and EXIT blocks
        self.entry_block = self.new_block()
        self.exit_block = self.new_block()

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

    def __str__(self):
        import json as _json

        return _json.dumps(self.to_dict(), indent=4)
