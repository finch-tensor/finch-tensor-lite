from typing import Any, Iterator
from abc import ABC, abstractmethod

"""
This module contains definitions for common functions that are useful for symbolic expression manipulation.
Its purpose is to provide a shared interface between various symbolic programming in Finch.

Classes:
    Term (ABC): An abstract base class representing a symbolic term. It provides methods to access the head
    of the term, its children, and to construct a new term with a similar structure.
"""


class Term(ABC):
    def __init__(self):
        self._hashcache = None  # Private field to cache the hash value

    @abstractmethod
    def head(self) -> Any:
        """Return the head type of the S-expression."""
        pass

    def children(self) -> list["Term"]:
        """Return the children (AKA tail) of the S-expression."""
        pass

    @abstractmethod
    def is_expr(self) -> bool:
        """Return True if the term is an expression tree, False otherwise. Must implement children() if True."""
        pass

    @abstractmethod
    def make_term(self, head: Any, children: list["Term"]) -> "Term":
        """
        Construct a new term in the same family of terms with the given head type and children.
        This function should satisfy `x == x.make_term(x.head(), *x.children())`
        """
        pass

    def __hash__(self) -> int:
        """Return the hash value of the term."""
        if self._hashcache is None:
            self._hashcache = hash(
                (0x1CA5C2ADCA744860, self.head(), tuple(self.children()))
            )
        return self._hashcache

    def __eq__(self, other: "Term") -> bool:
        self.head() == other.head() and self.children() == other.children()


def PostOrderDFS(node: Term) -> Iterator[Term]:
    if node.is_expr():
        for arg in node.children():
            yield from PostOrderDFS(arg)
    yield node


def PreOrderDFS(node: Term) -> Iterator[Term]:
    yield node
    if node.is_expr():
        for arg in node.children():
            yield from PreOrderDFS(arg)
