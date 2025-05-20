def intersect(x1: tuple, x2: tuple) -> tuple:
    return tuple(x for x in x1 if x in x2)


def is_subsequence(x1: tuple, x2: tuple) -> bool:
    return x1 == tuple(x for x in x2 if x in x1)


def with_subsequence(x1: tuple, x2: tuple) -> tuple:
    res = list(x2)
    indices = [idx for idx, val in enumerate(x2) if val in x1]
    for idx, i in enumerate(indices):
        res[i] = x1[idx]
    return tuple(res)
