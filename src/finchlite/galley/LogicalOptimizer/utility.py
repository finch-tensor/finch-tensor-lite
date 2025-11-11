from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import Any

from finchlite.finch_logic import LogicTree


def PreOrderDFS(
    roots: Iterable[Any] | Any,
    neighbors: Callable[[Any], Iterable[Any]],
    *,
    key: Callable[[Any], Hashable] | None = None,
) -> list[Any]:
    """Return nodes in depth-first PRE-ORDER."""
    if isinstance(roots, Iterable) and not isinstance(roots, (str, bytes)):
        root_iter: Iterable[Any] = roots
    else:
        root_iter = [roots]

    k = key or id
    visited: set[Hashable] = set()
    out: list[Any] = []

    for r in root_iter:
        kr = k(r)
        if kr in visited:
            continue
        visited.add(kr)
        out.append(r)
        stack: list[tuple[Any, Iterator[Any]]] = [(r, iter(neighbors(r)))]
        while stack:
            node, it = stack[-1]
            try:
                nxt = next(it)
            except StopIteration:
                stack.pop()
                continue
            kn = k(nxt)
            if kn in visited:
                continue
            visited.add(kn)
            out.append(nxt)
            stack.append((nxt, iter(neighbors(nxt))))
    return out


def PostOrderDFS(
    roots: Iterable[Any] | Any,
    neighbors: Callable[[Any], Iterable[Any]],
    *,
    key: Callable[[Any], Hashable] | None = None,
) -> list[Any]:
    """Return nodes in depth-first POST-ORDER."""
    if isinstance(roots, Iterable) and not isinstance(roots, (str, bytes)):
        root_iter: Iterable[Any] = roots
    else:
        root_iter = [roots]

    k = key or id
    visited: set[Hashable] = set()
    out: list[Any] = []

    for r in root_iter:
        kr = k(r)
        if kr in visited:
            continue
        stack: list[tuple[Any, Iterator[Any], bool]] = [(r, iter(neighbors(r)), False)]
        visited.add(kr)
        while stack:
            node, it, expanded = stack.pop()
            if not expanded:
                stack.append((node, it, True))
                for nxt in neighbors(node):
                    kn = k(nxt)
                    if kn in visited:
                        continue
                    visited.add(kn)
                    stack.append((nxt, iter(neighbors(nxt)), False))
            else:
                out.append(node)
    return out


def intree(n1, n2):
    """
    Return True iff `n1` occurs in the subtree rooted at `n2`.
    """
    target = n1
    stack = [n2]
    while stack:
        n = stack.pop()
        if n == target:
            return True
        if isinstance(n, LogicTree):
            stack.extend(n.children)
    return False


def isdescendant(n1, n2):
    """
    True iff `n1` is a strict descendant of `n2`.
    """
    if n1 == n2:
        return False
    return intree(n1, n2)
