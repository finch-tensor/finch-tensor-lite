from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Self

"""
This module contains definitions for common functions that are useful for symbolic
expression manipulation. Its purpose is to provide a shared interface between various
symbolic programming in Finch.

Classes:
    Term (ABC): An abstract base class representing a symbolic term. It provides methods
    to access the head of the term, its children, and to construct a new term with a
    similar structure.
"""


class Term:
    @abstractmethod
    def head(self) -> Any:
        """Return the head type of the S-expression."""

    @classmethod
    @abstractmethod
    def make_term(cls, head: Any, *children: Term) -> Self:
        """
        Construct a new term in the same family of terms with the given head type and
        children. This function should satisfy
        `x == x.make_term(x.head(), *x.children())`
        """


@dataclass(frozen=True, eq=True)
class TermTree(Term, ABC):
    @abstractmethod
    def children(self) -> list[Term]:
        """Return the children (AKA tail) of the S-expression."""


def PostOrderDFS(node: Term) -> Iterator[Term]:
    if isinstance(node, TermTree):
        for arg in node.children():
            yield from PostOrderDFS(arg)
    yield node


def PreOrderDFS(node: Term) -> Iterator[Term]:
    yield node
    if isinstance(node, TermTree):
        for arg in node.children():
            yield from PreOrderDFS(arg)
