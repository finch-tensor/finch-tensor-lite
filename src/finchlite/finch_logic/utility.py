from finchlite.finch_logic import LogicTree


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
