from collections.abc import Iterator

from finchlite.symbolic.term import Term, TermTree


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


def intree(n1, n2):
    """
    Return True iff `n1` occurs in the subtree rooted at `n2`.
    """
    return any(node == n1 for node in PostOrderDFS(n2))


def isdescendant(n1, n2):
    """
    True iff `n1` is a strict descendant of `n2`.
    """
    if n1 == n2:
        return False
    return intree(n1, n2)
