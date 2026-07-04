import math

import numpy as np

import finchlite
from finchlite.algebra import (
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
from finchlite.algebra.ftypes import FDType


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
    assert init_value(ffuncs.and_, finchlite.bool) is np.True_
    assert init_value(ffuncs.or_, finchlite.bool) is np.False_
    assert init_value(ffuncs.xor, finchlite.bool) is np.False_
    assert init_value(ffuncs.logaddexp, finchlite.float64) == -math.inf
    assert init_value(ffuncs.logical_and, finchlite.bool_) is True
    assert init_value(ffuncs.logical_or, finchlite.bool_) is False
    assert init_value(ffuncs.logical_xor, finchlite.bool_) is False
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
    assert promote_type(finchlite.bool, finchlite.bool_) == finchlite.bool
    assert promote_type(finchlite.bool_, finchlite.bool) == finchlite.bool
    assert promote_type(finchlite.int8, finchlite.int_) == finchlite.int8
    assert promote_type(finchlite.int_, finchlite.int8) == finchlite.int8
    assert promote_type(finchlite.int32, finchlite.int_) == finchlite.int32
    assert promote_type(finchlite.int_, finchlite.int32) == finchlite.int32
    assert promote_type(finchlite.uint8, finchlite.int_) == finchlite.uint8
    assert promote_type(finchlite.int_, finchlite.uint8) == finchlite.uint8
    assert promote_type(finchlite.int64, finchlite.bool_) == finchlite.int64
    assert promote_type(finchlite.bool_, finchlite.int64) == finchlite.int64
    assert promote_type(finchlite.float32, finchlite.int_) == finchlite.float32
    assert promote_type(finchlite.int_, finchlite.float32) == finchlite.float32
    assert promote_type(finchlite.float32, finchlite.float_) == finchlite.float32
    assert promote_type(finchlite.float_, finchlite.float32) == finchlite.float32
    assert promote_type(finchlite.complex64, finchlite.float_) == finchlite.complex64
    assert promote_type(finchlite.float_, finchlite.complex64) == finchlite.complex64
    assert promote_type(finchlite.complex64, finchlite.complex_) == finchlite.complex64
    assert promote_type(finchlite.complex_, finchlite.complex64) == finchlite.complex64
    tuple_type = TupleFType.from_tuple((finchlite.int32, finchlite.float32))
    promoted_tuple_type = TupleFType.from_tuple((finchlite.int64, finchlite.float32))
    assert isinstance(tuple_type, FDType)
    assert (
        promote_type(
            tuple_type,
            TupleFType.from_tuple((finchlite.int64, finchlite.int_)),
        )
        == promoted_tuple_type
    )
    assert (
        ffuncs.where.return_type(
            finchlite.bool,
            tuple_type,
            TupleFType.from_tuple((finchlite.int64, finchlite.int_)),
        )
        == promoted_tuple_type
    )
    assert (
        ffuncs.choose((0, 0)).return_type(
            tuple_type,
            TupleFType.from_tuple((finchlite.int64, finchlite.int_)),
        )
        == promoted_tuple_type
    )


def test_same_ffunc():
    assert ffuncs.same(1, 1)
    assert not ffuncs.same(1, 2)
    assert ffuncs.same(float("nan"), float("nan"))
    assert ffuncs.same(np.float32(np.nan), np.float64(np.nan))
    assert not ffuncs.same(float("nan"), 1.0)
    assert ffuncs.same(None, None)
    np.testing.assert_array_equal(
        ffuncs.same(np.array([1.0, np.nan, 2.0]), np.array([1.0, np.nan, np.nan])),
        np.array([True, True, False]),
    )
    assert not ffuncs.not_same(1, 1)
    assert ffuncs.not_same(1, 2)
    assert not ffuncs.not_same(float("nan"), float("nan"))
    assert not ffuncs.not_same(None, None)
    np.testing.assert_array_equal(
        ffuncs.not_same(np.array([1.0, np.nan, 2.0]), np.array([1.0, np.nan, np.nan])),
        np.array([False, False, True]),
    )


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
    assert ffuncs.samehash(np.float64(np.nan)) == ("nan", finchlite.float64)
    assert ffuncs.samehash(np.float32(np.nan)) == ("nan", finchlite.float32)
    assert ffuncs.samehash(SameHash()) == ("samehash", 1)
