from .nodes import (
    AssemblyNode,
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
    TaggedVariable,
    Unpack,
    WhileLoop,
)


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


class CFGBuilder:
    """Incrementally builds control-flow graphs for Finch Assembly IR."""

    def __init__(self):
        self.cfgs: dict[str, ControlFlowGraph] = {}
        self.current_block = None
        self.current_cfg: ControlFlowGraph

    def new_cfg(self, name: str) -> ControlFlowGraph:
        new_cfg = ControlFlowGraph(name)
        self.current_block = new_cfg.new_block()
        new_cfg.entry_block.add_successor(self.current_block)
        self.cfgs[name] = new_cfg
        return new_cfg

    def build(self, node: AssemblyNode):
        return self(node)

    def to_dict(self):
        """Convert all CFGs to dictionaries for JSON serialization."""
        return {cfg_name: cfg.to_dict() for cfg_name, cfg in self.cfgs.items()}

    def __str__(self):
        import json as _json

        return _json.dumps(self.to_dict(), indent=4)

    def __call__(self, node: AssemblyNode, break_block: BasicBlock | None = None):
        match node:
            case (
                Literal()
                | Unpack()
                | Repack()
                | Resize()
                | TaggedVariable()
                | GetAttr()
                | SetAttr()
                | Call()
                | Load()
                | Store()
                | Length()
                | Slot()
                | Stack()
                | Assign()
            ):
                self.current_block.add_statement(node)
            case Block(bodies):
                for body in bodies:
                    self(body, break_block)
            case If(cond, body):
                self(IfElse(cond, body, Block()))
            case IfElse(cond, body, else_body):
                before_block = self.current_block

                if_block = self.current_cfg.new_block()
                else_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                before_block.add_successor(if_block)
                before_block.add_successor(else_block)

                if_block.add_statement(cond)
                self.current_block = if_block
                self(body, break_block)
                self.current_block.add_successor(after_block)

                self.current_block = else_block
                self(else_body, break_block)
                self.current_block.add_successor(after_block)

                self.current_block = after_block
            case WhileLoop(cond, body):
                before_block = self.current_block

                body_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                before_block.add_successor(body_block)
                before_block.add_successor(after_block)

                body_block.add_statement(cond)
                self.current_block = body_block
                self(body, after_block)

                self.current_block.add_successor(before_block)
                self.current_block = after_block
            case ForLoop(var, _start, _var, body):
                before_block = self.current_block

                body_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                before_block.add_successor(body_block)
                before_block.add_successor(after_block)

                # TODO: figure out a RIGHT way to represent 'ForLoop' initialization
                # statement
                body_block.add_statement(Assign(var, var))
                self.current_block = body_block
                self(body, after_block)

                self.current_block.add_successor(body_block)
                self.current_block = after_block
            case BufferLoop(_buf, var, body):
                before_block = self.current_block

                body_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()

                before_block.add_successor(body_block)
                before_block.add_successor(after_block)

                # TODO: figure out a RIGHT way to represent 'BufferLoop' initialization
                # statement
                body_block.add_statement(Assign(var, var))
                self.current_block = body_block
                self(body, after_block)

                self.current_block.add_successor(body_block)
                self.current_block = after_block
            case Return(value):
                self.current_block.add_statement(Return(value))

                # when Return is met,
                # make a connection to the EXIT block of function (cfg)
                self.current_block.add_successor(
                    self.current_cfg.exit_block
                )

                # create a block where we going to store all unreachable statements
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
                            self.current_block.add_statement(arg)
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

                    func_name = (
                        func.name.variable.name
                        if hasattr(func.name, "variable")
                        else func.name.name
                    )

                    self.current_cfg = self.new_cfg(func_name)
                    self(func)
                    self.current_block.add_successor(
                        self.current_cfg.exit_block
                    )
            case node:
                raise NotImplementedError(node)

        return self.cfgs
