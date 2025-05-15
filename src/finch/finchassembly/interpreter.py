from .nodes import (
    Literal,
    Variable,
)


class FinchAssemblyInterpreter:
    def __init__(self):
        self.verbose = False
        self.bindings = {}
        self.parent = {}

    def __call__(self, node):
        if self.verbose:
            print(f"Evaluating: {node}")
        head = node.head()
        if head == Literal:
            return node.val
        elif head == Variable:
            if node in self.bindings:
                return self.bindings[node]
            else:
                raise ValueError(f"undefined variable {node.val}")
        # TODO not done
