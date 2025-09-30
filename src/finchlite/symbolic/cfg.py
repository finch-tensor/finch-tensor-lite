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

    def __str__(self) -> str:
        """String representation of BasicBlock in LLVM style."""
        lines = []
        
        if self.predecessors:
            pred_names = [pred.id for pred in self.predecessors]
            pred_str = f" #preds=[{', '.join(pred_names)}]"
        else:
            pred_str = " #preds=[]"
        
        # Block header
        lines.append(f"{self.id}:{pred_str}")
        
        # Block statements
        for stmt in self.statements:
            lines.append(f"    {stmt}")
        
        return "\n".join(lines)


class ControlFlowGraph:
    """Control-Flow Graph (CFG) for a single FinchAssembly function."""

    def __init__(self, func_name: str):
        self.block_counter = 0
        self.name = func_name
        self.blocks: dict[str, BasicBlock] = {}

        self.entry_block = self.new_block()
        self.exit_block = self.new_block()

    def new_block(self):
        bid = f"{self.name}_{self.block_counter}"
        self.block_counter += 1
        block = BasicBlock(bid)
        self.blocks[bid] = block
        return block

    def print(self) -> str:
        """Print the CFG in LLVM style format."""
        lines = []
        blocks = list(self.blocks.values())
        
        for i, block in enumerate(blocks):
            lines.append(str(block))
            
            # Add empty line between blocks (except for last block)
            if i < len(blocks) - 1:
                lines.append("")
        
        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation using LLVM-style pretty printing."""
        return self.print()


class CFGPrinterContext:
    def print(self, cfgs: dict) -> str:
        """Print multiple CFGs in LLVM style."""
        lines = []
        
        for cfg in cfgs.values():
            # CFG name and metadata (indent = 0)
            lines.append(f"{cfg.name}: #entry={cfg.entry_block.id}, #exit={cfg.exit_block.id}")
            
            # Get CFG string representation and indent it
            cfg_str = str(cfg)
            for line in cfg_str.split('\n'):
                if line.strip():  # Only indent non-empty lines
                    lines.append(f"    {line}")
                else:
                    lines.append("")
            
            # Add empty line between CFGs
            lines.append("")
        
        return "\n".join(lines).rstrip()