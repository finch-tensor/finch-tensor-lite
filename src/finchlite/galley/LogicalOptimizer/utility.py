from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Hashable, List, Optional, Set, Tuple

def PreOrderDFS(
    roots: Iterable[Any] | Any,
    neighbors: Callable[[Any], Iterable[Any]],
    *,
    key: Optional[Callable[[Any], Hashable]] = None,
) -> List[Any]:
    """Return nodes in depth-first PRE-ORDER."""
    if isinstance(roots, Iterable) and not isinstance(roots, (str, bytes)):
        root_iter: Iterable[Any] = roots
    else:
        root_iter = [roots]

    k = key or id
    visited: Set[Hashable] = set()
    out: List[Any] = []

    for r in root_iter:
        kr = k(r)
        if kr in visited:
            continue
        visited.add(kr)
        out.append(r)
        stack: List[Tuple[Any, Iterator[Any]]] = [(r, iter(neighbors(r)))]
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
    key: Optional[Callable[[Any], Hashable]] = None,
) -> List[Any]:
    """Return nodes in depth-first POST-ORDER."""
    if isinstance(roots, Iterable) and not isinstance(roots, (str, bytes)):
        root_iter: Iterable[Any] = roots
    else:
        root_iter = [roots]

    k = key or id
    visited: Set[Hashable] = set()
    out: List[Any] = []

    for r in root_iter:
        kr = k(r)
        if kr in visited:
            continue
        stack: List[Tuple[Any, Iterator[Any], bool]] = [(r, iter(neighbors(r)), False)]
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
