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
