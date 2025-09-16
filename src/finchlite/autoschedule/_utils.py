from collections.abc import Iterable, Sequence


def intersect(x1: Iterable, x2: Iterable) -> tuple:
    return tuple(x for x in x1 if x in x2)


def extend_uniqe(x1: Iterable, x2: Iterable) -> tuple:
    return tuple(x1) + setdiff(x2, x1)


def is_subsequence(x1: Iterable, x2: Iterable) -> bool:
    return tuple(x1) == tuple(x for x in x2 if x in x1)


def setdiff(x1: Iterable, x2: Iterable) -> tuple:
    return tuple([x for x in x1 if x not in x2])


def with_subsequence(x1: Sequence, x2: Iterable) -> tuple:
    res = list(x2)
    indices = [idx for idx, val in enumerate(x2) if val in x1]
    for idx, i in enumerate(indices):
        res[i] = x1[idx]
    return tuple(res)
