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

Notes:
    Although TermTree implements `children`, `make_term` is defined under Term.
    The reason for this is to enable writing IR-agnostic passes that operate on any kind
    of term, whether it is a leaf or an internal node. For example, a function that
    constructs or transforms a tree term may only have access to a leaf node, but still
    needs to call `make_term` on it.

    For example:

        def insert_wrapper(node: Term, pattern, wrap_head):
            if matches(node, pattern):
                return node.make_term(wrap_head, node)
            elseif isinstance(node, TermTree):
                def recurse(node: Term) -> Term:
                    return insert_wrapper(node, pattern, wrap_head)
                return node.make_term(node.head(), *(
                    recurse(child) for child in node.children
                ))
            else:
                return node

    This function would not be able to wrap leaf nodes if Term didn't define
    `make_term`.

    Also, `make_term` is not meant to be written differently for different
    members of Term.  Instead of overriding `make_term` in subclasses, introduce
    your own method to override, and call that from make_term.
"""


class Term:
    @abstractmethod
    def head(self) -> Any:
        """Return the head type of the S-expression."""
        ...

    @classmethod
    @abstractmethod
    def make_term(cls, head: Any, *children: Term) -> Self:
        """
        Construct a new term in the same family of terms with the given head type and
        children. This function should satisfy
        `x == x.make_term(x.head(), *x.children)`
        """
        ...


@dataclass(frozen=True, eq=True)
class TermTree(Term, ABC):
    @property
    @abstractmethod
    def children(self) -> list[Term]:
        """Return the children (AKA tail) of the S-expression."""
        ...


def PostOrderDFS(node: Term) -> Iterator[Term]:
    if isinstance(node, TermTree):
        for arg in node.children:
            yield from PostOrderDFS(arg)
    yield node


def PreOrderDFS(node: Term) -> Iterator[Term]:
    yield node
    if isinstance(node, TermTree):
        for arg in node.children:
            yield from PreOrderDFS(arg)
