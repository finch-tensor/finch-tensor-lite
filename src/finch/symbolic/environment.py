import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Generic, Optional, TypeVar

"""
    Namespace

A namespace for managing variable names and aesthetic fresh variable generation.
"""


class Namespace:
    def __init__(self):
        self.counts = defaultdict(int)
        self.resolutions = {}

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

    def resolve(self, *names: str):
        """
        Resolve a list of namespaced variable names to a unique name.
        e.g. `resolve("a", "b")` might return `a_b_1` if `a_b` has already been
        used in scope.
        """
        self.resolutions.setdefault(names, lambda: self.freshen("_".join(names)))


T = TypeVar("T")


class ScopedDict(Generic[T]):
    """
    A dictionary that allows for scoped variable resolution.
    """

    def __init__(
        self,
        bindings: dict[str, T] | None = None,
        parent: Optional["ScopedDict[T]"] = None,
    ):
        if bindings is None:
            bindings = {}
        self.bindings: dict[str, T] = bindings
        self.parent: ScopedDict[T] | None = parent

    def __getitem__(self, key: str) -> T:
        if key in self.bindings:
            return self.bindings[key]
        if self.parent is not None:
            return self.parent[key]
        raise KeyError(f"Key '{key}' not found in scoped dictionary.")

    def __setitem__(self, key: str, value: T) -> None:
        self.bindings[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.bindings or (self.parent is not None and key in self.parent)

    def scope(self) -> "ScopedDict[T]":
        """
        Create a new scoped dictionary that inherits from this one.
        """
        return ScopedDict(parent=self)

    def get(self, key: str, default: T | None = None) -> T | None:
        """
        Get the value for a key, returning default if not found.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: str, default: T) -> T:
        """
        Set the default value for a key if it does not exist.
        """
        if key not in self.bindings:
            self.bindings[key] = default
        return self.bindings[key]


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

    def resolve(self, *names: str):
        return self.namespace.resolve(*names)

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
