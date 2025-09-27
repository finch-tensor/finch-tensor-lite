import json
from typing import Dict, Any


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


class CFGPrinterContext:
    def __init__(self, indent: int = 4):
        self.indent = indent
    
    def print(self, cfgs: dict) -> str:
        all_cfgs = {}
        
        for cfg in cfgs.values():
            all_cfgs[cfg.name] = self.cfg_to_dict(cfg)
        return json.dumps(all_cfgs, indent=self.indent, ensure_ascii=False)
    
    def cfg_to_dict(self, cfg: ControlFlowGraph) -> Dict[str, Any]:
        blocks_dict = {}
        
        # Convert each block to dictionary format
        for block_id, block in cfg.blocks.items():
            blocks_dict[block_id] = self.block_to_dict(block)
        
        return {
            "name": cfg.name,
            "entry_block": cfg.entry_block.id,
            "exit_block": cfg.exit_block.id,
            "blocks": blocks_dict
        }
    
    def block_to_dict(self, block: BasicBlock) -> Dict[str, Any]:
        return {
            "id": block.id,
            "statements": [str(stmt) for stmt in block.statements],
            "successors": [succ.id for succ in block.successors],
            "predecessors": [pred.id for pred in block.predecessors]
        }
