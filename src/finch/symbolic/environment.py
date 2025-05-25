import re
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

"""
    Namespace

A namespace for managing variable names and aesthetic fresh variable generation.
"""


class Namespace:
    def __init__(self):
        self.counts = defaultdict(int)

    def freshen(self, *tags):
        name = "_".join(str(tag) for tag in tags)
        m = re.match(r"^(.*)_(\d*)$", name)
        if m is None:
            tag = name
            n = 1
        else:
            tag = m.group(1)
            n = int(m.group(2))
        n = max(self.counts[tag] + 1, n)
        self.counts[tag] = n
        if n == 1:
            return tag
        return f"{tag}_{n}"


"""
    AbstractContext

A context for compiling code, managing side effects, and
variable names in the generated code of the executing environment.
"""


class AbstractContext(ABC):
    def __init__(self, namespace=None, preamble=None, epilogue=None):
        self.namespace = namespace if namespace is not None else Namespace()
        self.preamble = preamble if preamble is not None else []
        self.epilogue = epilogue if epilogue is not None else []

    def exec(self, thunk: Any):
        self.preamble.append(thunk)

    def post(self, thunk: Any):
        self.epilogue.append(thunk)

    def freshen(self, *tags):
        return self.namespace.freshen(*tags)

    def block(self):
        """
        Create a new block. Preambles and epilogues will stay within this block.
        """
        blk = self.__class__()
        blk.namespace = self.namespace
        blk.preamble = []
        blk.epilogue = []
        return blk

    @abstractmethod
    def emit(self):
        """
        Emit the code in this context.
        """


class AbstractSymbolic(ABC):
    """
    Abstract base class for symbolic objects. Symbolic objects are used to
    represent objects that are defined with respect to the state inside a
    symbolic computation context.
    """

    def foo(x):
        x + 1
