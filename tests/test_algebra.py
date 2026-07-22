import math

import numpy as np

import finch
from finch.algebra import (
    TupleFType,
    cansplitpush,
    ffuncs,
    init_value,
    is_annihilator,
    is_associative,
    is_distributive,
    is_idempotent,
    is_identity,
    promote_type,
    repeat_operator,
)
from finch.algebra.ftypes import FDType


def test_algebra_selected():
    assert is_distributive(ffuncs.mul, ffuncs.add)
    assert is_distributive(ffuncs.mul, ffuncs.sub)
    assert is_distributive(ffuncs.and_, ffuncs.or_)
    assert is_distributive(ffuncs.and_, ffuncs.xor)
    assert is_distributive(ffuncs.or_, ffuncs.and_)
    assert is_distributive(ffuncs.logical_and, ffuncs.logical_or)
    assert is_distributive(ffuncs.logical_and, ffuncs.logical_xor)
    assert is_distributive(ffuncs.logical_or, ffuncs.logical_and)
    assert is_annihilator(ffuncs.add, math.inf)
    assert is_annihilator(ffuncs.mul, 0)
    assert is_annihilator(ffuncs.or_, True)
    assert is_annihilator(ffuncs.and_, False)
    assert is_annihilator(ffuncs.logaddexp, math.inf)
    assert is_annihilator(ffuncs.logical_or, True)
    assert is_annihilator(ffuncs.logical_and, False)
    assert is_identity(ffuncs.add, 0)
    assert is_identity(ffuncs.mul, 1)
    assert is_identity(ffuncs.or_, False)
    assert is_identity(ffuncs.and_, True)
    assert is_identity(ffuncs.truediv, 1)
    assert is_identity(ffuncs.lshift, 0)
    assert is_identity(ffuncs.rshift, 0)
    assert is_identity(ffuncs.pow, 1)
    assert is_identity(ffuncs.truediv, 1)
    assert is_identity(ffuncs.logaddexp, -math.inf)
    assert is_identity(ffuncs.logical_or, False)
    assert is_identity(ffuncs.logical_and, True)
    assert is_identity(ffuncs.max, -math.inf)
    assert is_identity(ffuncs.min, math.inf)
    assert is_identity(ffuncs.choose(0), 0)
    assert is_associative(ffuncs.add)
    assert is_associative(ffuncs.mul)
    assert is_associative(ffuncs.choose(0))
    assert is_associative(ffuncs.logical_and)
    assert is_associative(ffuncs.logical_xor)
    assert is_associative(ffuncs.logical_or)
    assert is_associative(ffuncs.logaddexp)
    assert init_value(ffuncs.and_, finch.bool) is np.True_
    assert init_value(ffuncs.or_, finch.bool) is np.False_
    assert init_value(ffuncs.xor, finch.bool) is np.False_
    assert init_value(ffuncs.logaddexp, finch.float64) == -math.inf
    assert init_value(ffuncs.logical_and, finch.bool_) is True
    assert init_value(ffuncs.logical_or, finch.bool_) is False
    assert init_value(ffuncs.logical_xor, finch.bool_) is False
    assert is_idempotent(ffuncs.and_)
    assert is_idempotent(ffuncs.or_)
    assert is_idempotent(ffuncs.logical_and)
    assert is_idempotent(ffuncs.logical_or)
    assert is_idempotent(ffuncs.min)
    assert is_idempotent(ffuncs.max)
    assert is_idempotent(ffuncs.minby)
    assert is_idempotent(ffuncs.maxby)
    assert is_idempotent(ffuncs.add) is False
    assert is_idempotent(ffuncs.mul) is False
    assert is_idempotent(ffuncs.xor) is False
    assert is_idempotent(ffuncs.logical_xor) is False
    assert is_idempotent(ffuncs.logaddexp) is False
    assert repeat_operator(ffuncs.add) is ffuncs.mul
    assert repeat_operator(ffuncs.mul) is ffuncs.pow
    assert repeat_operator(ffuncs.and_) is None
    assert cansplitpush(ffuncs.add, ffuncs.add) is True
    assert cansplitpush(ffuncs.add, ffuncs.mul) is False
    assert cansplitpush(ffuncs.and_, ffuncs.and_) is False
    assert ffuncs.choose(0)(0, 2, 3) == 2
    assert ffuncs.choose(0)(0, 0) == 0
    assert ffuncs.choose(np.nan)(np.nan, 4.0) == 4.0
    assert ffuncs.minby((1, 10), (2, 20)) == (1, 10)
    assert ffuncs.minby((2, 10), (1, 20)) == (1, 20)
    assert ffuncs.minby((1, 10), (1, 20)) == (1, 10)
    assert ffuncs.maxby((2, 10), (1, 20)) == (2, 10)
    assert ffuncs.maxby((1, 10), (2, 20)) == (2, 20)
    assert ffuncs.maxby((1, 10), (1, 20)) == (1, 20)
    assert ffuncs.last((1, 2, 3)) == 3
    assert ffuncs.scaled_power(2.0) is ffuncs.scaled_square
    assert ffuncs.add_scaled_power(2.0) is ffuncs.add_scaled_square
    assert ffuncs.root_scaled_power(2.0) is ffuncs.root_scaled_square
    assert ffuncs.scaled_square(np.float64(0.0)) == (np.float64(0.0), 0.0)
    assert ffuncs.scaled_square(np.float64(3.0)) == (np.float64(1.0), 3.0)
    scaled_sum = ffuncs.add_scaled_square((1.0, 3.0), (1.0, 4.0))
    assert scaled_sum == (1.5625, 4.0)
    assert ffuncs.root_scaled_square(scaled_sum) == 5.0
    scaled_square_nan = ffuncs.add_scaled_square((0.0, 0.0), (1.0, math.nan))
    assert math.isnan(scaled_square_nan[0])
    assert math.isnan(scaled_square_nan[1])
    scaled_power_nan = ffuncs.add_scaled_power(3.0)((0.0, 0.0), (1.0, math.nan))
    assert math.isnan(scaled_power_nan[0])
    assert math.isnan(scaled_power_nan[1])
    scaled_negative_zero = ffuncs.scaled_negative_power(-2.0)(np.float64(0.0))
    assert math.isinf(scaled_negative_zero[0])
    assert scaled_negative_zero[1] == 0.0
    scaled_negative_sum = ffuncs.add_scaled_negative_power(-2.0)((1.0, 2.0), (1.0, 4.0))
    assert scaled_negative_sum == (1.25, 2.0)
    assert math.isclose(
        ffuncs.root_scaled_negative_power(-2.0)(scaled_negative_sum),
        2.0 / math.sqrt(1.25),
    )
    scaled_negative_nan = ffuncs.add_scaled_negative_power(-2.0)(
        (1.0, 2.0), (1.0, math.nan)
    )
    assert math.isnan(scaled_negative_nan[0])
    assert math.isnan(scaled_negative_nan[1])
    assert (
        ffuncs.root_scaled_negative_power(-2.0)(
            ffuncs.add_scaled_negative_power(-2.0)((1.0, 2.0), (math.inf, 0.0))
        )
        == 0.0
    )


def test_python_scalar_promotion_uses_weak_bottom():
    assert promote_type(finch.bool, finch.bool_) == finch.bool
    assert promote_type(finch.bool_, finch.bool) == finch.bool
    assert promote_type(finch.int8, finch.int_) == finch.int8
    assert promote_type(finch.int_, finch.int8) == finch.int8
    assert promote_type(finch.int32, finch.int_) == finch.int32
    assert promote_type(finch.int_, finch.int32) == finch.int32
    assert promote_type(finch.uint8, finch.int_) == finch.uint8
    assert promote_type(finch.int_, finch.uint8) == finch.uint8
    assert promote_type(finch.int64, finch.bool_) == finch.int64
    assert promote_type(finch.bool_, finch.int64) == finch.int64
    assert promote_type(finch.float32, finch.int_) == finch.float32
    assert promote_type(finch.int_, finch.float32) == finch.float32
    assert promote_type(finch.float32, finch.float_) == finch.float32
    assert promote_type(finch.float_, finch.float32) == finch.float32
    assert promote_type(finch.complex64, finch.float_) == finch.complex64
    assert promote_type(finch.float_, finch.complex64) == finch.complex64
    assert promote_type(finch.complex64, finch.complex_) == finch.complex64
    assert promote_type(finch.complex_, finch.complex64) == finch.complex64
    tuple_type = TupleFType.from_tuple((finch.int32, finch.float32))
    promoted_tuple_type = TupleFType.from_tuple((finch.int64, finch.float32))
    assert isinstance(tuple_type, FDType)
    assert (
        promote_type(
            tuple_type,
            TupleFType.from_tuple((finch.int64, finch.int_)),
        )
        == promoted_tuple_type
    )
    assert (
        ffuncs.where.return_type(
            finch.bool,
            tuple_type,
            TupleFType.from_tuple((finch.int64, finch.int_)),
        )
        == promoted_tuple_type
    )
    assert (
        ffuncs.choose((0, 0)).return_type(
            tuple_type,
            TupleFType.from_tuple((finch.int64, finch.int_)),
        )
        == promoted_tuple_type
    )


def test_ftype_recognizes_numpy_dtype_aliases():
    int_long = finch.int32 if np.dtype(np.long) == np.dtype(np.int32) else finch.int64
    uint_long = (
        finch.uint32 if np.dtype(np.ulong) == np.dtype(np.uint32) else finch.uint64
    )
    uintp = finch.uint32 if np.uintp == np.uint32 else finch.uint64
    cases = [
        (np.long, int_long),
        (np.ulong, uint_long),
        (np.intp, finch.intp),
        (np.uintp, uintp),
        (np.longlong, finch.int64),
        (np.ulonglong, finch.uint64),
        (np.float16, finch.float16),
    ]

    for np_type, finch_type in cases:
        assert finch.ftype(np_type) == finch_type
        assert finch.ftype(np_type(1)) == finch_type
        assert finch.ftype(np.dtype(np_type)) == finch_type


def test_floor_divide_return_type_handles_all_integer_dtypes():
    dtypes = [
        finch.bool,
        finch.int8,
        finch.int16,
        finch.int32,
        finch.int64,
        finch.uint8,
        finch.uint16,
        finch.uint32,
        finch.uint64,
    ]

    for x1 in dtypes:
        for x2 in dtypes:
            assert isinstance(ffuncs.floordiv.return_type(x1, x2), FDType)


def test_same_ffunc():
    assert ffuncs.same(1, 1)
    assert not ffuncs.same(1, 2)
    assert ffuncs.same(float("nan"), float("nan"))
    assert ffuncs.same(np.float32(np.nan), np.float64(np.nan))
    assert not ffuncs.same(float("nan"), 1.0)
    assert ffuncs.same(None, None)
    assert not ffuncs.not_same(1, 1)
    assert ffuncs.not_same(1, 2)
    assert not ffuncs.not_same(float("nan"), float("nan"))
    assert not ffuncs.not_same(None, None)


def test_same_ffunc_dunder_overload():
    class LeftSame:
        def __same__(self, other):
            return np.False_

    class RightSame:
        def __rsame__(self, other):
            return np.True_

    class LeftDefers:
        def __same__(self, other):
            return NotImplemented

    assert ffuncs.same(LeftSame(), RightSame()) is np.False_
    assert ffuncs.same(LeftDefers(), RightSame()) is np.True_


def test_samehash():
    class SameHash:
        def __samehash__(self):
            return ("samehash", 1)

    assert ffuncs.samehash(1) == 1
    assert ffuncs.samehash(np.float64(np.nan)) == ("nan", finch.float64)
    assert ffuncs.samehash(np.float32(np.nan)) == ("nan", finch.float32)
    assert ffuncs.samehash(SameHash()) == ("samehash", 1)
