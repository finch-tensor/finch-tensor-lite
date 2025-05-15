"""
This module contains definitions for common functions that are useful for symbolic
expression manipulation. Its purpose is to provide a shared interface between various
symbolic programming in Finch.

Classes:
    Term (ABC): An abstract base class representing a symbolic term. It provides methods
    to access the head of the term, its children, and to construct a new term with a
    similar structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Self

__all__ = ["Term", "PreOrderDFS", "PostOrderDFS"]


class Term(ABC):
    def __init__(self):
        self._hashcache = None  # Private field to cache the hash value

    @classmethod
    def head(cls) -> Callable[..., Self]:
        """Return the head type of the S-expression."""
        raise NotImplementedError

    @abstractmethod
    def children(self) -> list[Term]:
        """Return the children (AKA tail) of the S-expression."""

    @abstractmethod
    def is_expr(self) -> bool:
        """
        Return True if the term is an expression tree, False otherwise. Must implement
        `children()` if `True`."""

    @classmethod
    def make_term(cls, head: Callable[..., Self], *children: Term) -> Self:
        """
        Construct a new term in the same family of terms with the given
        children. This function should satisfy
        `x == x.make_term(x.head(), *x.children())`
        """
        raise NotImplementedError

    def __hash__(self) -> int:
        """Return the hash value of the term."""
        if self._hashcache is None:
            self._hashcache = hash(
                (0x1CA5C2ADCA744860, self.head(), tuple(self.children()))
            )
        return self._hashcache

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Term):
            return NotImplemented
        return self.head() is other.head() and self.children() == other.children()


def PostOrderDFS(node: Term) -> Iterator[Term]:
    if node.is_expr():
        arg: Term
        for arg in node.children():
            yield from PostOrderDFS(arg)
    yield node


def PreOrderDFS(node: Term) -> Iterator[Term]:
    yield node
    if node.is_expr():
        arg: Term
        for arg in node.children():
            yield from PreOrderDFS(arg)
