from enum import Enum
from typing import Any

from .. import finch_notation as ntn
from .lower import Extent

# TODO: move all extent classes and functions here


class _CombineStyle(Enum):
    UNION = 1
    INTERSECT = 2


def _combine_extents(ext_1: Extent, ext_2: Extent, style: _CombineStyle) -> Extent:
    if style == _CombineStyle.UNION:
        start_fn, end_fn = min, max
    else:
        start_fn, end_fn = max, min

    start_1, start_2 = ext_1.ftype.get_start(ext_1), ext_2.ftype.get_start(ext_2)
    end_1, end_2 = ext_1.ftype.get_end(ext_1), ext_2.ftype.get_end(ext_2)
    return Extent(
        start=ntn.Call(ntn.Literal(start_fn), (start_1, end_1)),
        end=ntn.Call(ntn.Literal(end_fn), (start_2, end_2)),
    )


def intersect_extents(ext_1: Extent, ext_2: Extent) -> Extent:
    return _combine_extents(ext_1, ext_2, _CombineStyle.INTERSECT)


def union_extents(ext_1: Extent, ext_2: Extent) -> Extent:
    return _combine_extents(ext_1, ext_2, _CombineStyle.UNION)


def get_start(ext: Extent) -> Any:
    return ext.ftype.get_start(ext)


def get_end(ext: Extent) -> Any:
    return ext.ftype.get_end(ext)


def get_unit(ext: Extent) -> Any:
    return ext.ftype.get_unit(ext)
