"""
This module provides a set of classes and utilities for symbolic term rewriting.
Rewriters transform terms based on specific rules.  A rewriter is any callable
which takes a Term and returns a Term or `None`.  A rewriter can return `None`
if there are no changes applicable to the input Term.  The module includes
various strategies for applying rewriters, such as recursive rewriting, chaining
multiple rewriters, and caching results.

Classes:
    Rewrite: A wrapper for a rewriter function that ensures the original term is
        returned if the rewriter produces no changes.
    PreWalk: Recursively rewrites each node in a term using a pre-order traversal.
    PostWalk: Recursively rewrites each node in a term using a post-order traversal.
    Chain: Applies a sequence of rewriters to a term, stopping when a rewriter
        produces a change.
    Fixpoint: Repeatedly applies a rewriter to a term until no further changes
        are made.
    Prestep: Recursively rewrites each node in a term, stopping if the rewriter
        produces no changes.
    Memo: Caches the results of a rewriter to avoid redundant computations.
"""

from collections.abc import Callable, Iterable
from typing import TypeVar

from .term import Term, TermTree

T = TypeVar("T", bound="Term")

RwCallable = Callable[[T], T | None]


def default_rewrite(x: T | None, y: T) -> T:
    return x if x is not None else y


class Rewrite:
    """
    A rewriter which returns the original argument even if `rw` returns nothing.

    Attributes:
        rw (RwCallable): The rewriter function to apply.
    """

    def __init__(self, rw: RwCallable):
        self.rw = rw

    def __call__(self, x: T) -> T:
        return default_rewrite(self.rw(x), x)


class PreWalk:
    """
    A rewriter which recursively rewrites each node using `rw`, then rewrites
    the arguments of the resulting node. If all rewriters return `nothing`,
    returns `nothing`.

    Attributes:
        rw (RwCallable): The rewriter function to apply.
    """

    def __init__(self, rw: RwCallable):
        self.rw = rw

    def __call__(self, x: T) -> T | None:
        y = self.rw(x)
        if y is not None:
            if isinstance(y, TermTree):
                args = y.children
                return y.make_term(  # type: ignore[return-value]
                    y.head(), *[default_rewrite(self(arg), arg) for arg in args]
                )
            return y
        if isinstance(x, TermTree):
            args = x.children
            new_args = list(map(self, args))
            if not all(arg is None for arg in new_args):
                return x.make_term(  # type: ignore[return-value]
                    x.head(),
                    *map(lambda x1, x2: default_rewrite(x1, x2), new_args, args),
                )
        return None


class PostWalk:
    """
    A rewriter which recursively rewrites the arguments of each node using
    `rw`, then rewrites the resulting node. If all rewriters return `nothing`,
    returns `nothing`.

    Attributes:
        rw (RwCallable): The rewriter function to apply.
    """

    def __init__(self, rw: RwCallable):
        self.rw = rw

    def __call__(self, x: T) -> T | None:
        if isinstance(x, TermTree):
            args = x.children
            new_args = list(map(self, args))
            if all(arg is None for arg in new_args):
                return self.rw(x)
            y = x.make_term(
                x.head(), *map(lambda x1, x2: default_rewrite(x1, x2), new_args, args)
            )
            return default_rewrite(self.rw(y), y)  # type: ignore[return-value]
        return self.rw(x)


class Chain:
    """
    A rewriter which rewrites using each rewriter in `itr`. If all rewriters
    return `nothing`, return `nothing`.

    Attributes:
        rws (Iterable[RwCallable]): A collection of rewriter functions to apply.
    """

    def __init__(self, rws: Iterable[RwCallable]):
        self.rws = rws

    def __call__(self, x: T) -> T | None:
        is_success = False
        for rw in self.rws:
            y = rw(x)
            if y is not None:
                is_success = True
                x = y
        if is_success:
            return x
        return None


class Fixpoint:
    """
    A rewriter which repeatedly applies `rw` to `x` until no changes are made. If
    the rewriter first returns `nothing`, returns `nothing`.

    Attributes:
        rw (RwCallable): The rewriter function to apply.
    """

    def __init__(self, rw: RwCallable):
        self.rw = rw

    def __call__(self, x: T) -> T | None:
        y = self.rw(x)
        if y is not None:
            while y is not None and x != y:
                x = y
                y = self.rw(x)
            return x
        return None


class Prestep:
    """
    A rewriter which recursively rewrites each node using `rw`. If `rw` is
    nothing, it returns `nothing`, otherwise it recurses to the arguments.

    Attributes:
        rw (RwCallable): The rewriter function to apply.
    """

    def __init__(self, rw: RwCallable):
        self.rw = rw

    def __call__(self, x: T) -> T | None:
        y = self.rw(x)
        if y is not None and isinstance(y, TermTree):
            y_args = y.children
            return y.make_term(
                y.head(), *[default_rewrite(self(arg), arg) for arg in y_args]
            )
        return y


class Memo:
    """
    A rewriter which caches the results of `rw` in `cache` and returns the
    result.

    Attributes:
        rw (RwCallable): The rewriter function to apply.
        cache (dict): A dictionary to store cached results.
    """

    def __init__(self, rw: RwCallable, cache: dict | None = None):
        self.rw = rw
        self.cache = cache if cache is not None else {}

    def __call__(self, x: T) -> T | None:
        if x not in self.cache:
            self.cache[x] = self.rw(x)
        return self.cache[x]
