from collections.abc import Iterable

from .nodes import Field


class NonConcordantLists(Exception):
    pass


def merge_concordant(args: Iterable[Iterable[Field]]) -> list[Field]:
    merge_list: list[Field] = []
    visited: set[Field] = set()

    for arg in args:
        idx = 0
        for f in arg:
            if f not in visited:
                visited.add(f)
                merge_list.insert(idx, f)
                idx += 1
            else:
                next_idx = merge_list.index(f)
                if next_idx < idx:
                    raise NonConcordantLists
                idx = next_idx + 1

    return merge_list
