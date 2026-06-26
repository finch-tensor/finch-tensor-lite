from enum import Enum

from finchlite import finch_notation as ntn
from finchlite.algebra import ffuncs

from .lower import SymbolicExtent

# TODO: move all extent classes and functions here


class _CombineStyle(Enum):
    UNION = 1
    INTERSECT = 2


def _combine_extents(
    ext_1: SymbolicExtent, ext_2: SymbolicExtent, style: _CombineStyle
) -> SymbolicExtent:
    start_fn: ffuncs.FinchOperator
    end_fn: ffuncs.FinchOperator
    if style == _CombineStyle.UNION:
        start_fn, end_fn = ffuncs.min, ffuncs.max
    else:
        start_fn, end_fn = ffuncs.max, ffuncs.min

    start_1, start_2 = ext_1.get_start(), ext_2.get_start()
    end_1, end_2 = ext_1.get_end(), ext_2.get_end()
    return SymbolicExtent(
        start_sym=ntn.Call(ntn.Literal(start_fn), (start_1, start_2)),
        end_sym=ntn.Call(ntn.Literal(end_fn), (end_1, end_2)),
    )


def intersect_extents(ext_1: SymbolicExtent, ext_2: SymbolicExtent) -> SymbolicExtent:
    return _combine_extents(ext_1, ext_2, _CombineStyle.INTERSECT)


def union_extents(ext_1: SymbolicExtent, ext_2: SymbolicExtent) -> SymbolicExtent:
    return _combine_extents(ext_1, ext_2, _CombineStyle.UNION)
