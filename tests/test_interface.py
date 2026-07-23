import dataclasses
import importlib
import math
import warnings

import pytest

import numpy as np
import scipy.sparse as scipy_sparse

import finch
from finch import ffuncs
from finch.algebra import ftype
from finch.finch_logic import MapJoin, Query, Reorder

from .conftest import finch_assert_allclose, finch_assert_equal


# Utility function to generate random complex numpy tensors
def random_array(shape, dtype=np.complex128, rng: np.random.Generator | None = None):
    """Generates a random complex array. Uses integers for both real
    and imaginary parts to avoid floating-point issues in tests.

    Args:
        - shape: A tuple specifying the shape of the array.
        - dtype: The intended dtype of the randomly generated array.
        If nothing is provided, np.complex128 array is generated
        - rng: The random number generator to use. Providing one is strongly recommended
        for reproducibility. If no generator is provided, a new one is instantiated with
        random seed.

    Returns:
        A NumPy array of complex numbers with the given shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    if dtype is bool:
        arr = rng.integers(0, 2, size=shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        arr = rng.integers(-100, 100, size=shape).astype(dtype)
    elif np.issubdtype(dtype, complex):
        real = rng.random(size=shape).astype(np.float32)
        imag = rng.random(size=shape).astype(np.float32)
        arr = (real + 1j * imag).astype(dtype)
    else:
        arr = rng.random(size=shape).astype(dtype)
    return arr


@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
        (np.array([[2, 0], [1, 3]]), np.array([[4, 1], [2, 2]])),
    ],
)
def test_matrix_multiplication(a, b):
    result = finch.fuse(
        lambda a, b: finch.reduce(
            ffuncs.add, finch.multiply(finch.expand_dims(a, 2), b), axis=1
        ),
        a,
        b,
    )

    expected = np.matmul(a, b)

    finch_assert_equal(result, expected)


class TestOverrideTensorFType(finch.TensorFType):
    # This class doesn't define any pytests
    __test__ = False

    def __init__(self, fmt):
        self.fmt = fmt

    def __eq__(self, other):
        if not isinstance(other, TestOverrideTensorFType):
            return False
        return self.fmt == other.fmt

    def __hash__(self):
        return hash(self.fmt)

    def construct(self, shape: tuple):
        return TestOverrideTensor(
            np.full(shape, self.fmt.fill_value, dtype=self.fmt.element_type)
        )

    def __call__(self, val):
        """
        Convert a tensor to this test eager tensor type.

        Args:
            val: A value to convert to this type.
        Returns:
            A TestOverrideTensor instance of this type.
        """
        raise NotImplementedError(
            f"Tensor conversion not yet implemented for {type(self).__name__}"
        )

    def from_numpy(self, arr):
        return TestOverrideTensor(arr)

    @property
    def fill_value(self):
        return self.fmt.fill_value

    @property
    def element_type(self):
        return self.fmt.element_type

    @property
    def shape_type(self):
        return self.fmt.shape_type


class TestOverrideTensor(finch.OverrideTensor):
    # This class doesn't define any pytests
    __test__ = False

    def __init__(self, array):
        self.array = finch.asarray(array)

    def __repr__(self):
        return f"TestOverrideTensor({self.array})"

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, idx, val):
        self.array[idx] = val

    def item(self):
        return self.array.item()

    def __hash__(self):
        # TODO: correct hashing for ndarrays
        return id(self.array)

    @property
    def shape(self):
        return self.array.shape

    @property
    def ftype(self):
        return TestOverrideTensorFType(finch.ftype(self.array))

    @property
    def fill_value(self):
        return self.ftype.fill_value

    @property
    def element_type(self):
        return self.ftype.element_type

    @property
    def shape_type(self):
        return self.ftype.shape_type

    def to_numpy(self):
        return self.array

    def to_scipy(self):
        raise NotImplementedError(f"{type(self).__name__} does not support to_scipy.")


@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
        (np.array([[2, 0], [1, 3]]), np.array([[4, 1], [2, 2]])),
        (np.array([[2, 0], [1, 3]]), 2),
        (3, np.array([[2, 0], [1, 3]])),
        (np.array([[2, 0], [1, 3]]), True),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "b_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "ops, np_op",
    [
        ((finch.add, np.add), np.add),
        ((finch.subtract, np.subtract), np.subtract),
        ((finch.multiply, np.multiply), np.multiply),
        ((finch.bitwise_and, np.bitwise_and), np.bitwise_and),
        ((finch.bitwise_or, np.bitwise_or), np.bitwise_or),
        ((finch.bitwise_xor, np.bitwise_xor), np.bitwise_xor),
        (
            (finch.bitwise_left_shift, np.bitwise_left_shift),
            np.bitwise_left_shift,
        ),
        (
            (finch.bitwise_right_shift, np.bitwise_right_shift),
            np.bitwise_right_shift,
        ),
        (
            (
                finch.truediv,
                np.true_divide,
                finch.divide,
                np.divide,
            ),
            np.true_divide,
        ),
        ((finch.floor_divide, np.floor_divide), np.floor_divide),
        ((finch.mod, np.mod), np.mod),
        ((finch.power, np.power), np.power),
        ((finch.mod, np.mod, finch.remainder, np.remainder), np.mod),
        ((finch.pow, np.pow), np.pow),
        ((finch.hypot, np.hypot), np.hypot),
        ((finch.atan2, np.atan2), np.atan2),
        ((finch.logaddexp, np.logaddexp), np.logaddexp),
        ((finch.copysign, np.copysign), np.copysign),
        ((finch.nextafter, np.nextafter), np.nextafter),
        ((finch.logical_and, np.logical_and), np.logical_and),
        ((finch.logical_or, np.logical_or), np.logical_or),
        ((finch.logical_xor, np.logical_xor), np.logical_xor),
        ((finch.minimum, np.minimum), np.minimum),
        ((finch.maximum, np.maximum), np.maximum),
        ((finch.equal, np.equal), np.equal),
        ((finch.not_equal, np.not_equal), np.not_equal),
        ((finch.less, np.less), np.less),
        ((finch.less_equal, np.less_equal), np.less_equal),
        ((finch.greater, np.greater), np.greater),
        ((finch.greater_equal, np.greater_equal), np.greater_equal),
    ],
)
def test_elementwise_operations(a, b, a_wrap, b_wrap, ops, np_op):
    wa = a_wrap(a)
    wb = b_wrap(b)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in",
        )

        expected = np_op(a, b)

        for op in ops:
            result = op(wa, wb)

            if isinstance(wa, finch.LazyTensor) or isinstance(wb, finch.LazyTensor):
                assert isinstance(result, finch.LazyTensor)

                result = finch.compute(result)

            finch_assert_equal(result, expected)


@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        (-np.inf, 1.0, -np.inf),
        (-np.inf, -1.0, np.inf),
        (np.inf, -1.0, -np.inf),
        (np.inf, 1.0, np.inf),
        (-1.0, np.inf, -0.0),
        (1.0, -np.inf, -0.0),
    ],
)
def test_floor_divide_float_special_cases(x1, x2, expected):
    x1 = finch.asarray(x1, dtype=finch.float64)
    x2 = finch.asarray(x2, dtype=finch.float64)

    for result in (finch.floor_divide(x1, x2), x1 // x2):
        result = float(result)
        assert result == expected
        assert np.signbit(result) == np.signbit(expected)


@pytest.mark.parametrize(
    "x1, x2, expected",
    [
        (-np.inf, 1.0, -np.inf),
        (-np.inf, -1.0, np.inf),
        (np.inf, -1.0, -np.inf),
        (np.inf, 1.0, np.inf),
        (-1.0, np.inf, -0.0),
        (1.0, -np.inf, -0.0),
    ],
)
def test_ffunc_floor_divide_float_special_cases(x1, x2, expected):
    result = float(ffuncs.floordiv(np.float64(x1), np.float64(x2)))
    assert result == expected
    assert np.signbit(result) == np.signbit(expected)


@pytest.mark.parametrize("wrap", [lambda x: x, finch.lazy])
def test_same_elementwise_nan(wrap):
    a = np.array([1.0, np.nan, np.nan, 2.0])
    b = np.array([1.0, np.nan, 0.0, np.nan])

    same = finch.same(wrap(a), wrap(b))
    not_same = finch.not_same(wrap(a), wrap(b))

    if isinstance(same, finch.LazyTensor):
        same = finch.compute(same)
    if isinstance(not_same, finch.LazyTensor):
        not_same = finch.compute(not_same)

    expected = np.array([True, True, False, False])
    finch_assert_equal(same, expected)
    finch_assert_equal(not_same, np.logical_not(expected))


@pytest.mark.parametrize("wrap", [lambda x: x, finch.lazy])
def test_count_nonfill(wrap):
    x = np.array([[0.0, 1.0, np.nan], [2.0, 0.0, 0.0]])

    count = finch.count_nonfill(wrap(x))
    axis_count = finch.count_nonfill(wrap(x), axis=1)

    if isinstance(count, finch.LazyTensor):
        count = finch.compute(count)
    if isinstance(axis_count, finch.LazyTensor):
        axis_count = finch.compute(axis_count)

    finch_assert_equal(count, np.array(3))
    finch_assert_equal(axis_count, np.array([2, 1]))


def test_count_nonfill_nan_fill_value():
    x = finch.full((2, 3), np.nan)
    assert np.isnan(x.fill_value)
    finch_assert_equal(finch.count_nonfill(x), np.array(0))


def test_array_api_constants():
    assert finch.e == math.e
    assert finch.pi == math.pi
    assert finch.inf == math.inf
    assert math.isnan(finch.nan)
    assert finch.nan != finch.nan
    assert finch.newaxis is None

    assert bool(finch.isinf(finch.asarray(finch.inf)))
    assert bool(finch.isnan(finch.asarray(finch.nan)))


def test_asarray_python_scalars_use_default_array_dtypes():
    scalar = finch.asarray(1)

    assert finch.__array_api_version__ == "2024.12"
    assert finch.asarray(True).dtype == finch.bool
    assert scalar.dtype == finch.int64
    assert finch.asarray(1.0).dtype == finch.float64
    assert finch.asarray(1j).dtype == finch.complex128
    assert finch.asarray(1.0, dtype=finch.float32).dtype == finch.float32
    assert scalar.__array_namespace__() is finch


def test_asarray_existing_finch_tensors_pass_through():
    scalar = finch.asarray(1)
    lazy = finch.lazy(1)

    assert finch.asarray(scalar) is scalar
    assert finch.asarray(lazy) is lazy
    assert scalar.dtype == finch.int64


def test_array_namespace_info():
    info = finch.__array_namespace_info__()

    assert info.capabilities() == {
        "boolean indexing": False,
        "data-dependent shapes": False,
        "max dimensions": 5,
    }
    assert info.default_device() == finch.serial()
    assert info.devices() == [finch.serial(), finch.cpu()]
    assert info.default_dtypes() == {
        "real floating": finch.float64,
        "complex floating": finch.complex128,
        "integral": finch.int64,
        "indexing": finch.intp,
    }
    assert info.dtypes(kind="bool") == {"bool": finch.bool}
    assert info.dtypes(kind="integral") == {
        "int8": finch.int8,
        "int16": finch.int16,
        "int32": finch.int32,
        "int64": finch.int64,
        "uint8": finch.uint8,
        "uint16": finch.uint16,
        "uint32": finch.uint32,
        "uint64": finch.uint64,
    }
    assert set(info.dtypes(kind=("real floating", "complex floating"))) == {
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    }

    with pytest.raises(ValueError):
        info.default_dtypes(device="gpu")


def test_array_object_metadata():
    x = finch.asarray(np.arange(6).reshape(2, 3))

    assert x.device == finch.serial()
    assert x.size == 6
    assert x.to_device(None) is x
    assert x.to_device(finch.serial()) is x
    finch_assert_equal(x.T, x.to_numpy().T)

    cpu_dev = finch.cpu("test", n=2)
    x_cpu = x.to_device(cpu_dev)
    assert x_cpu.device == cpu_dev
    assert x_cpu is not x
    assert finch.asarray(np.arange(3), device=cpu_dev).device == cpu_dev

    y = finch.asarray(np.arange(24).reshape(2, 3, 4))
    finch_assert_equal(y.mT, np.swapaxes(y.to_numpy(), -1, -2))
    with pytest.raises(ValueError):
        _ = y.T
    with pytest.raises(ValueError):
        x.to_device("gpu")


def test_device_hierarchy_objects_and_ftypes():
    ser = finch.serial()
    assert isinstance(ser, finch.AbstractDevice)
    assert finch.ftype(ser) == finch.SerialFType()
    assert ser.num_tasks == 1
    assert ser.device == ser
    assert ser.parent_device is None
    assert finch.SerialFType().device == finch.SerialFType()

    cpu_dev = finch.cpu("main", n=3)
    assert isinstance(cpu_dev, finch.CPU)
    assert cpu_dev == finch.CPU(finch.serial(), id="main", n=7)
    assert cpu_dev.num_tasks == 3
    assert cpu_dev.device == cpu_dev
    assert cpu_dev.parent_device == finch.serial()
    assert finch.ftype(cpu_dev) == finch.CPUFType("main")
    assert finch.CPUFType("main").device == finch.CPUFType("main")
    assert finch.CPUFType("main").parent_device_type == finch.SerialFType()
    assert finch.CPUFType("main")(2) == finch.CPU(finch.serial(), id="main", n=2)
    assert finch.common_device(finch.serial(), cpu_dev) == cpu_dev
    assert finch.common_device(cpu_dev, finch.serial()) == cpu_dev
    with pytest.raises(ValueError):
        finch.common_device(cpu_dev, finch.cpu("other", n=3))

    parent = finch.SerialTask()
    thread = finch.CPUThread(2, cpu_dev, parent)
    thread_type = finch.CPUThreadFType(finch.ftype(parent), cpu_dev.ftype)
    assert thread.device == cpu_dev
    assert finch.ftype(thread) == thread_type
    assert thread_type.device == cpu_dev.ftype
    assert thread.parent_task == parent
    assert thread_type.parent_task == finch.ftype(parent)
    assert thread.task_num == 2
    assert thread.is_on_device(cpu_dev)
    assert finch.is_on_device(thread, cpu_dev)


def test_finfo_returns_python_scalars():
    info = finch.finfo(finch.float32)
    property_info = finch.float32.finfo

    assert dataclasses.is_dataclass(info)
    assert isinstance(info, finch.FInfo)
    assert info == property_info
    assert isinstance(info.bits, int)
    assert isinstance(info.eps, float)
    assert isinstance(info.max, float)
    assert isinstance(info.min, float)
    assert isinstance(info.smallest_normal, float)
    assert info.dtype == finch.float32


def test_iinfo_returns_python_scalars():
    info = finch.iinfo(finch.int16)
    property_info = finch.int16.iinfo

    assert dataclasses.is_dataclass(info)
    assert isinstance(info, finch.IInfo)
    assert info == property_info
    assert isinstance(info.bits, int)
    assert isinstance(info.max, int)
    assert isinstance(info.min, int)
    assert info.dtype == finch.int16


def test_result_type():
    assert finch.result_type(finch.int8, finch.int16) == finch.int16
    assert finch.result_type(finch.int32, finch.uint32) == finch.int64
    assert finch.result_type(finch.float32, finch.float64) == finch.float64
    assert finch.result_type(finch.complex64, finch.complex128) == finch.complex128
    assert (
        finch.result_type(finch.asarray([1], dtype=finch.int32), finch.uint16)
        == finch.int32
    )


def test_result_type_python_scalars_are_weak():
    assert finch.result_type(finch.bool, True) == finch.bool
    assert finch.result_type(finch.int32, 1) == finch.int32
    assert finch.result_type(finch.float32, 1, 1.0) == finch.float32
    assert finch.result_type(finch.complex64, 1, 1.0, 1j) == finch.complex64

    with pytest.raises(TypeError):
        finch.result_type(1, 1.0)


def test_lazy_python_scalars_keep_builtin_dtypes():
    assert finch.lazy(True).dtype == finch.bool_
    assert finch.lazy(1).dtype == finch.int_
    assert finch.lazy(1.0).dtype == finch.float_
    assert finch.lazy(1j).dtype == finch.complex_


def test_nan_fill_value_ftype_equality():
    x = finch.full((2, 3), np.nan)
    y = finch.full((2, 3), np.nan)
    assert finch.same(x.fill_value, y.fill_value)
    assert x.ftype == y.ftype
    assert hash(x.ftype) == hash(y.ftype)

    lazy_x = finch.lazy(x)
    lazy_y = finch.lazy(y)
    assert lazy_x.ftype == lazy_y.ftype
    assert hash(lazy_x.ftype) == hash(lazy_y.ftype)

    scalar_x = finch.asarray(finch.nan)
    scalar_y = finch.asarray(finch.nan)
    assert scalar_x.ftype == scalar_y.ftype
    assert hash(scalar_x.ftype) == hash(scalar_y.ftype)


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[1, 2], [3, 4]])),
        (np.array([[2, 0], [1, 3]])),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "ops, np_op",
    [
        ((finch.abs, np.abs), np.abs),
        ((finch.positive, np.positive), np.positive),
        ((finch.negative, np.negative), np.negative),
        ((finch.bitwise_invert, np.bitwise_invert), np.bitwise_invert),
        ((finch.reciprocal, np.reciprocal), np.reciprocal),
        ((finch.sin, np.sin), np.sin),
        ((finch.sinh, np.sinh), np.sinh),
        ((finch.cos, np.cos), np.cos),
        ((finch.cosh, np.cosh), np.cosh),
        ((finch.tan, np.tan), np.tan),
        ((finch.tanh, np.tanh), np.tanh),
        ((finch.asin, np.asin), np.asin),
        ((finch.asinh, np.asinh), np.asinh),
        ((finch.acos, np.acos), np.acos),
        ((finch.acosh, np.acosh), np.acosh),
        ((finch.atan, np.atan), np.atan),
        ((finch.atanh, np.atanh), np.atanh),
        ((finch.round, np.round), np.round),
        ((finch.floor, np.floor), np.floor),
        ((finch.ceil, np.ceil), np.ceil),
        ((finch.trunc, np.trunc), np.trunc),
        ((finch.exp, np.exp), np.exp),
        ((finch.expm1, np.expm1), np.expm1),
        ((finch.log, np.log), np.log),
        ((finch.log1p, np.log1p), np.log1p),
        ((finch.log2, np.log2), np.log2),
        ((finch.log10, np.log10), np.log10),
        ((finch.signbit, np.signbit), np.signbit),
        ((finch.sqrt, np.sqrt), np.sqrt),
        ((finch.square, np.square), np.square),
        ((finch.sign, np.sign), np.sign),
        ((finch.isfinite, np.isfinite), np.isfinite),
        ((finch.isinf, np.isinf), np.isinf),
        ((finch.isnan, np.isnan), np.isnan),
        ((finch.logical_not, np.logical_not), np.logical_not),
    ],
)
def test_unary_operations(a, a_wrap, ops, np_op):
    wa = a_wrap(a)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in",
        )

        expected = np_op(a)

        for op in ops:
            result = op(wa)

            if isinstance(wa, finch.LazyTensor):
                assert isinstance(result, finch.LazyTensor)

                result = finch.compute(result)

            finch_assert_equal(result, expected)


@pytest.mark.parametrize(
    "a",
    [
        np.array([1 + 2j, 3 - 4j, 0 + 1j]),
        np.array([[1j, -1j], [2 + 3j, -4 - 5j]]),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "op, np_op",
    [
        (finch.real, np.real),
        (finch.imag, np.imag),
    ],
)
def test_complex_operations(a, a_wrap, op, np_op):
    wa = a_wrap(a)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in",
        )

        expected = np_op(a)
        result = op(wa)

        if isinstance(wa, finch.LazyTensor):
            assert isinstance(result, finch.LazyTensor)

            result = finch.compute(result)

        finch_assert_equal(result, expected)


@pytest.mark.parametrize(
    "a, b, c",
    [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 1], [1, 1]]),
            np.array([[3, 3], [3, 3]]),
        ),
        (
            np.array([[2, -1], [0, 5]]),
            None,
            np.array([[1, 1], [1, 1]]),
        ),
        (
            np.array([[0, -3], [5, 10]]),
            np.array([[0, 0], [0, 0]]),
            None,
        ),
        (
            np.array([[-5, 0], [10, 7]]),
            -2,
            2,
        ),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "b_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "c_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "ops, np_op, caller",
    [
        ((finch.clip, np.clip), np.clip, lambda op, a, b, c: op(a, min=b, max=c)),
    ],
)
def test_ternary_operations(a, b, c, a_wrap, b_wrap, c_wrap, ops, np_op, caller):
    wa = a_wrap(a)
    wb = b_wrap(b)
    wc = c_wrap(c)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in",
        )

        expected = np_op(a, b, c)

        for op in ops:
            result = caller(op, wa, wb, wc)

            if (
                isinstance(wa, finch.LazyTensor)
                or isinstance(wb, finch.LazyTensor)
                or isinstance(wc, finch.LazyTensor)
            ):
                assert isinstance(result, finch.LazyTensor)

                result = finch.compute(result)

            finch_assert_equal(result, expected)


@pytest.mark.parametrize(
    "a",
    [
        (np.array([[True, False, True, False], [False, False, False, False]])),
        (np.array([[1, 2], [3, 4]])),
        (np.array([[2, 0], [1, 3]])),
        (np.array([[1.00002, -12.618, 0, 0.001], [-1.414, -5.01, 0, 0]])),
        (np.array([[0, 0.618, 0, 0.001], [0, 0.01, 0, 0]])),
        (np.array([[10000.0, 1.0, -89.0, 78], [401.0, 3, 5, 10.2]])),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "op, np_op",
    [
        (finch.prod, lambda x, axis: np.prod(x, axis=axis, dtype=x.dtype)),
        (finch.sum, lambda x, axis: np.sum(x, axis=axis, dtype=x.dtype)),
        (finch.any, np.any),
        (finch.all, np.all),
        (finch.min, np.min),
        (finch.max, np.max),
        (finch.mean, np.mean),
        (finch.std, np.std),
        (finch.var, np.var),
    ],
)
@pytest.mark.parametrize(
    "axis",
    [
        None,
        0,
        1,
        (0, 1),
    ],
)
def test_reduction_operations(a, a_wrap, op, np_op, axis):
    wa = a_wrap(a)

    if a.dtype == np.bool_ and np_op in (np.mean, np.std, np.var):
        pytest.skip("Boolean arrays do not support mean, std, var operations")

    expected = np_op(a, axis=axis)

    result = op(wa, axis=axis)

    if isinstance(wa, finch.LazyTensor):
        assert isinstance(result, finch.LazyTensor)

        result = finch.compute(result)

    if np.issubdtype(expected.dtype, np.floating) or np.issubdtype(
        expected.dtype, np.complexfloating
    ):
        finch_assert_allclose(result, expected, rtol=1e-15, atol=0.0)
    else:
        finch_assert_equal(result, expected)


@pytest.mark.usefixtures("interpreter_scheduler")
def test_std_nan_propagation():
    x = np.array([np.nan], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        result = finch.compute(finch.std(finch.lazy(x)))
    assert np.isnan(result.item())


@pytest.mark.parametrize("wrap", [lambda x: x, TestOverrideTensor, finch.lazy])
@pytest.mark.parametrize(
    "op, np_op", [(finch.argmin, np.argmin), (finch.argmax, np.argmax)]
)
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_argmin_argmax(wrap, op, np_op, axis, keepdims):
    x = np.array([[4.0, 1.0, 1.0], [2.0, 2.0, 3.0]])
    result = op(wrap(x), axis=axis, keepdims=keepdims)
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)

    assert result.dtype in (finch.int32, finch.int64)
    finch_assert_equal(result, np_op(x, axis=axis, keepdims=keepdims))


@pytest.mark.parametrize("wrap", [lambda x: x, TestOverrideTensor, finch.lazy])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_argsort(wrap, axis, descending):
    x = np.array(
        [
            [3.0, 1.0, 2.0, 1.0, 5.0],
            [0.0, 4.0, 4.0, -1.0, 2.0],
            [2.0, 4.0, 3.0, -1.0, 0.0],
        ]
    )
    key = -x if descending else x
    expected_indices = np.argsort(key, axis=axis, kind="stable")
    expected_sorted = np.take_along_axis(x, expected_indices, axis=axis)

    indices = finch.argsort(
        wrap(x),
        axis=axis,
        descending=descending,
        stable=False,
    )
    sorted_x = finch.sort(
        wrap(x),
        axis=axis,
        descending=descending,
    )
    if isinstance(indices, finch.LazyTensor):
        indices = finch.compute(indices)
    if isinstance(sorted_x, finch.LazyTensor):
        sorted_x = finch.compute(sorted_x)

    assert indices.dtype in (finch.int32, finch.int64)
    finch_assert_equal(indices, expected_indices)
    finch_assert_equal(sorted_x, expected_sorted)


def test_argsort_stable_ties():
    x = np.array([[2, 1, 2, 1, 2]], dtype=np.int64)

    finch_assert_equal(finch.argsort(x), np.array([[1, 3, 0, 2, 4]]))
    finch_assert_equal(
        finch.argsort(x, descending=True),
        np.array([[0, 2, 4, 1, 3]]),
    )


@pytest.mark.parametrize("wrap", [lambda x: x, TestOverrideTensor, finch.lazy])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("use_sorter", [False, True])
def test_searchsorted(wrap, side, use_sorter):
    sorted_x1 = np.array([1.0, 2.0, 2.0, 4.0, 7.0])
    x2 = np.array([[0.0, 2.0, 3.0], [7.0, 8.0, 1.0]])
    if use_sorter:
        x1 = np.array([4.0, 1.0, 7.0, 2.0, 2.0])
        sorter = np.argsort(x1, kind="stable")
        expected = np.searchsorted(x1, x2, side=side, sorter=sorter)
        result = finch.searchsorted(
            wrap(x1),
            wrap(x2),
            side=side,
            sorter=wrap(sorter),
        )
    else:
        expected = np.searchsorted(sorted_x1, x2, side=side)
        result = finch.searchsorted(wrap(sorted_x1), wrap(x2), side=side)

    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)

    assert result.dtype in (finch.int32, finch.int64)
    finch_assert_equal(result, expected)


@pytest.mark.parametrize("wrap", [lambda x: x, finch.lazy])
@pytest.mark.parametrize("op, np_op", [(finch.min, np.min), (finch.max, np.max)])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_min_max_nan_propagation(wrap, op, np_op, axis):
    x = np.array([[1.0, np.nan], [3.0, 4.0]])
    result = op(wrap(x), axis=axis)
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)
    finch_assert_equal(result, np_op(x, axis=axis))


@pytest.mark.parametrize("op", [finch.minimum, finch.maximum])
@pytest.mark.parametrize("wrap", [lambda x: x, finch.lazy])
def test_minimum_maximum_python_scalar_promotion(wrap, op):
    x = np.array([1.0, 2.0], dtype=np.float32)
    result = op(wrap(x), 1.0)
    assert result.dtype == finch.float32
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)
    assert result.dtype == finch.float32


@pytest.mark.parametrize(
    "a, b",
    [
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ),
        (
            np.arange(12, dtype=np.float64).reshape(3, 4),
            np.arange(8, dtype=np.float64).reshape(4, 2),
        ),
    ],
)
def test_matmul_bufferized_ndarray(a, b):
    ba = finch.asarray(a)
    bb = finch.asarray(b)
    expected = a @ b

    result = finch.matmul(ba, bb)
    result_with_op = ba @ bb

    finch_assert_allclose(result, expected)
    finch_assert_allclose(result_with_op, expected)


@pytest.mark.usefixtures(
    "interpreter_scheduler"
)  # batched and broadcasting not supported
@pytest.mark.parametrize(
    "a, b",
    [
        # 1D x 1D (dot product)
        (np.array([1, 2, 3]), np.array([4, 5, 6])),
        # 2D x 2D
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])),
        # 2D x 1D
        (np.array([[1, 2], [3, 4]]), np.array([5, 6])),
        # 1D x 2D
        (np.array([1, 2]), np.array([[3, 4], [5, 6]])),
        # 3D x 3D (batched matmul)
        (
            np.arange(2 * 3 * 4).reshape(2, 3, 4),
            np.arange(2 * 4 * 5).reshape(2, 4, 5),
        ),
        # Broadcasting cases
        # 1D x 2D (broadcasting)
        (np.array([1, 2]), np.arange(2 * 4).reshape(2, 4)),
        # 1D x 3D (broadcasting)
        (np.array([1, 2]), np.arange(3 * 2 * 5).reshape(3, 2, 5)),
        #  4D x 3D (broadcasting)
        (
            np.arange(7 * 2 * 4 * 3).reshape(7, 2, 4, 3),
            np.arange(2 * 3 * 4).reshape(2, 3, 4),
        ),
        # 3D x 1D (broadcasting)
        (np.arange(3 * 2 * 4).reshape(3, 2, 4), np.arange(4)),
        # (1, 3, 2) x (5, 2, 3)
        (np.arange(1 * 3 * 2).reshape(1, 3, 2), np.arange(5 * 2 * 3).reshape(5, 2, 3)),
        # Complex numbers, 4D x 5D
        (
            random_array((2, 3, 4, 5)),
            random_array((3, 5, 6)),
        ),
        # mismatch dimensions
        (
            np.arange(7 * 2 * 3 * 4).reshape(7, 2, 3, 4),
            np.arange(2 * 3 * 4).reshape(2, 3, 4),
        ),
        (np.arange(5), np.arange(4)),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "b_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
def test_matmul(a, b, a_wrap, b_wrap):
    """
    Tests for matrix multiplication using finch's matmul function.
    """
    wa = a_wrap(a)
    wb = b_wrap(b)

    try:
        expected = np.linalg.matmul(a, b)
    except ValueError:
        with pytest.raises(ValueError):
            finch.matmul(wa, wb)  # make sure matmul raises error too
            _ = wa @ wb
        return

    result = finch.matmul(wa, wb)
    result_with_op = wa @ wb  # make sure the operator overload works too
    result_with_np = np.matmul(wa, wb)

    if isinstance(wa, finch.LazyTensor) or isinstance(wb, finch.LazyTensor):
        assert isinstance(result, finch.LazyTensor)
        result = finch.compute(result)
        result_with_op = finch.compute(result_with_op)
        result_with_np = finch.compute(result_with_np)

    assert finch.ftype(expected.dtype.type) == result.element_type, (
        f"Expected dtype {expected.dtype}, got {result.dtype}"
    )
    finch_assert_allclose(result, expected)
    finch_assert_allclose(result_with_op, expected)
    finch_assert_allclose(result_with_np, expected)


def test_outer_default_scheduler():
    a = np.array([1, 2])
    b = np.array([3, 4, 5])

    result = finch.outer(a, b)

    finch_assert_equal(result, np.outer(a, b))


def test_linalg_outer_eager():
    a = finch.asarray(np.array([1, 2]))
    b = finch.asarray(np.array([3, 4, 5]))

    result = finch.linalg.outer(a, b)

    assert not isinstance(result, finch.LazyTensor)
    finch_assert_equal(result, np.outer(a.to_numpy(), b.to_numpy()))


@pytest.mark.usefixtures("interpreter_scheduler")
@pytest.mark.parametrize(
    "a, b",
    [
        (np.array([1, 2, 3]), np.array([4, 5])),
        (np.array([1.5, 2.0]), np.array([-1.0, 3.0, 4.0])),
        (random_array((2,)), random_array((3,))),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "b_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
def test_outer(a, b, a_wrap, b_wrap):
    wa = a_wrap(a)
    wb = b_wrap(b)
    expected = np.outer(a, b)

    result = finch.outer(wa, wb)
    result_with_np = np.outer(wa, wb)

    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)
    if isinstance(result_with_np, finch.LazyTensor):
        result_with_np = finch.compute(result_with_np)

    finch_assert_allclose(result, expected)
    finch_assert_allclose(result_with_np, expected)


def test_outer_uses_single_logic_query():
    a = finch.lazy(np.array([1, 2]))
    b = finch.lazy(np.array([3, 4, 5]))

    result = finch.outer(a, b)
    queries = [stmt for stmt in result.ctx.trace() if isinstance(stmt, Query)]

    assert result.shape == (2, 3)
    assert len(queries) == 3
    assert isinstance(queries[-1].rhs, Reorder)
    assert isinstance(queries[-1].rhs.arg, MapJoin)
    assert queries[-1].rhs.arg.op.val == ffuncs.mul


@pytest.mark.parametrize(
    "a, b",
    [
        (np.ones((2, 2)), np.ones(3)),
        (np.ones(2), np.ones((3, 1))),
    ],
)
def test_outer_requires_vectors(a, b):
    with pytest.raises(ValueError):
        finch.outer(a, b)


@pytest.mark.parametrize(
    "a, n",
    [
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 2),
        (np.arange(9, dtype=np.float64).reshape(3, 3), 3),
    ],
)
def test_matrix_power_bufferized_ndarray(a, n):
    ba = finch.asarray(a)
    expected = np.linalg.matrix_power(a, n)

    result = finch.linalg.matrix_power(ba, n)

    finch_assert_allclose(result, expected)


@pytest.mark.usefixtures("interpreter_scheduler")
@pytest.mark.parametrize(
    "a, n",
    [
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 0),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 1),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 2),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 5),
        (np.array([[2.0, 0.0], [0.0, 3.0]]), 4),
        (np.arange(9, dtype=np.float64).reshape(3, 3), 3),
        (np.stack([np.array([[1.0, 2.0], [3.0, 4.0]]), np.eye(2)]), 3),
        # invalid: non-square
        (np.ones((2, 3)), 2),
        # invalid: 1D
        (np.ones((3,)), 2),
    ],
)
@pytest.mark.parametrize(
    "wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
def test_matrix_power(a, n, wrap):
    wa = wrap(a)

    try:
        expected = np.linalg.matrix_power(a, n)
    except (ValueError, np.linalg.LinAlgError):
        with pytest.raises(ValueError):
            result = finch.linalg.matrix_power(wa, n)
            if isinstance(result, finch.LazyTensor):
                finch.compute(result)
        return

    result = finch.linalg.matrix_power(wa, n)

    if isinstance(result, finch.LazyTensor):
        assert isinstance(wa, finch.LazyTensor)
        result = finch.compute(result)

    assert finch.ftype(expected.dtype.type) == result.element_type
    finch_assert_allclose(result, expected)


def test_matrix_power_negative_eager():
    a = np.array([[1.0, 2.0], [3.0, 5.0]])
    expected = np.linalg.matrix_power(a, -3)

    result = finch.linalg.matrix_power(a, -3)

    finch_assert_allclose(result, expected)


def test_matrix_power_negative_lazy_requires_materialization():
    a = finch.lazy(np.array([[1.0, 2.0], [3.0, 5.0]]))
    expected = np.linalg.matrix_power(np.array([[1.0, 2.0], [3.0, 5.0]]), -1)

    with pytest.warns(RuntimeWarning, match="matrix_power|inv"):
        result = finch.linalg.matrix_power(a, -1)

    finch_assert_allclose(result, expected)


@pytest.mark.parametrize(
    "a, n",
    [
        (np.ones((2, 2)), 1.5),
    ],
)
def test_matrix_power_invalid_n(a, n):
    with pytest.raises((ValueError, TypeError)):
        finch.linalg.matrix_power(a, n)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "a",
    [
        np.arange(6).reshape(2, 3),
        np.arange(12).reshape(1, 12),
        np.arange(24).reshape(2, 3, 4),  # 3D array
        # Complex
        random_array((5, 1, 4)),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
def test_matrix_transpose(a, a_wrap):
    """
    Tests for matrix transpose
    """
    a = np.array(a)
    wa = a_wrap(a)
    expected = np.linalg.matrix_transpose(a)

    result = finch.matrix_transpose(wa)
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)
    finch_assert_equal(result, expected)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "a, b, axes",
    [
        # 1D x 1D (dot product)
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 0),
        # axes as int
        (np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4), 1),
        (np.arange(24).reshape(2, 3, 4), np.arange(12).reshape(4, 3), 1),
        (np.arange(24).reshape(2, 4, 3), np.arange(24).reshape(4, 3, 2), 2),
        # axes as tuple of sequences
        (
            np.arange(24).reshape(2, 3, 4),
            np.arange(24).reshape(4, 3, 2),
            ([1, 2], [1, 0]),
        ),
        (
            np.arange(60).reshape(3, 4, 5),
            np.arange(24).reshape(4, 3, 2),
            ([0, 1], [1, 0]),
        ),
        # axes=0 (outer product)
        (np.arange(3), np.arange(4), 0),
        (np.arange(8 * 7 * 5).reshape(8, 7, 5), np.arange(12).reshape(3, 4, 1), 0),
        (np.arange(3, dtype=np.uint8), np.arange(4, dtype=np.uint8), ((), ())),
        # complex
        (random_array((2, 3)), random_array((3, 4)), 1),
        (
            random_array((3, 5, 4, 6)),
            random_array((6, 4, 5, 3)),
            ([2, 1, 3], [1, 2, 0]),
        ),
        # mismatched axes (should raise)
        (np.arange(6).reshape(2, 3), np.arange(8).reshape(2, 4), 1),
    ],
)
@pytest.mark.parametrize(
    "a_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "b_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
def test_tensordot(a, b, axes, a_wrap, b_wrap):
    """
    Tests for tensordot operation according to the Array API specification.
    See: https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.tensordot.html
    """
    wa = a_wrap(a)
    wb = b_wrap(b)
    try:
        expected = np.tensordot(a, b, axes=axes)
    except ValueError:
        # tensordot should raise a ValueError
        with pytest.raises(ValueError):
            finch.tensordot(wa, wb, axes=axes)
        return
    result = finch.tensordot(wa, wb, axes=axes)

    if isinstance(wa, finch.LazyTensor) or isinstance(wb, finch.LazyTensor):
        assert isinstance(result, finch.LazyTensor)
        result = finch.compute(result)
    finch_assert_allclose(result, expected)
    assert result.to_numpy().dtype == expected.dtype


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
def test_tensordot_default_axes():
    a = np.arange(24).reshape(2, 3, 4)
    b = np.arange(24).reshape(3, 4, 2)

    result = finch.tensordot(a, b)

    finch_assert_allclose(result, np.tensordot(a, b))


@pytest.mark.parametrize(
    "x1, x2, axis",
    [
        # 1D x 1D (scalar result)
        (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), -1),
        # 2D x 2D (vector result)
        (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]), -1),
        # 3D x 3D, axis=-1
        (
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            -1,
        ),
        # 3D x 3D, axis=-2
        (
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            -2,
        ),
        # Broadcasting: (2, 3, 4) x (4,)
        (
            np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4),
            np.arange(4, dtype=float),
            -1,
        ),
        # Broadcasting: (3, 4) x (1, 4)
        (
            np.arange(3 * 4, dtype=float).reshape(3, 4),
            np.arange(4, dtype=float).reshape(1, 4),
            -1,
        ),
        # Complex numbers
        (np.array([1 + 2j, 3 + 4j]), np.array([5 - 1j, 2 + 2j]), -1),
        # axis=0
        (
            np.arange(2 * 3, dtype=float).reshape(2, 3),
            np.arange(2 * 3, dtype=float).reshape(2, 3),
            0,
        ),
        # Mismatched contracted axis
        (np.ones((2, 3)), np.ones((2, 4)), -1),
        (np.ones((5,)), np.ones((6,)), -1),
        # Broadcasting not allowed on contracted axis
        (np.ones((2, 3)), np.ones((1, 3)), 0),
    ],
)
@pytest.mark.parametrize(
    "x1_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "x2_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
def test_vecdot(x1, x2, axis, x1_wrap, x2_wrap):
    """
    Tests for vector dot product operation according to the Array API specification.
    See: https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.vecdot.html
    """
    wx1 = x1_wrap(x1)
    wx2 = x2_wrap(x2)
    try:
        expected = np.linalg.vecdot(x1, x2, axis=axis)
    except ValueError:
        with pytest.raises(ValueError):
            finch.vecdot(wx1, wx2, axis=axis)
        return

    result = finch.vecdot(wx1, wx2, axis=axis)
    if isinstance(wx1, finch.LazyTensor) or isinstance(wx2, finch.LazyTensor):
        assert isinstance(result, finch.LazyTensor)
        result = finch.compute(result)

    finch_assert_allclose(result, expected)


def test_vecdot_preserves_promoted_input_dtype():
    x1 = finch.asarray(np.array([1, 2, 3], dtype=np.uint8))
    x2 = finch.asarray(np.array([4, 5, 6], dtype=np.uint8))

    result = finch.vecdot(x1, x2)

    assert result.element_type == finch.uint8
    assert result.fill_value == np.uint8(0)
    assert result.item() == np.uint8(32)


@pytest.mark.usefixtures("interpreter_scheduler")
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"axis": 1},
        {"axis": 0, "keepdims": True, "ord": 1},
        {"axis": (0, 1), "ord": 0},
        {"axis": 1, "ord": float("inf")},
        {"axis": 1, "ord": -float("inf")},
        {"axis": None, "ord": -2},
    ],
)
@pytest.mark.parametrize(
    "wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
def test_linalg_vector_norm(kw, wrap):
    x = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float64)
    wx = wrap(x)
    expected = np.linalg.vector_norm(x, **kw)

    result = finch.linalg.vector_norm(wx, **kw)
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)

    assert finch.ftype(expected.dtype.type) == result.element_type
    finch_assert_allclose(result, expected)


@pytest.mark.usefixtures("interpreter_scheduler")
def test_linalg_vector_norm_large_values():
    x = np.array([1e308, 1e308], dtype=np.float64)
    expected = np.float64(math.sqrt(2.0) * 1e308)

    result = finch.compute(finch.linalg.vector_norm(finch.lazy(x)))

    assert np.isfinite(result.item())
    finch_assert_allclose(result, expected)


@pytest.mark.usefixtures("interpreter_scheduler")
def test_linalg_vector_norm_negative_ord_stable_values():
    tiny = np.array([1e-308, 1e308], dtype=np.float64)
    tiny_result = finch.compute(finch.linalg.vector_norm(finch.lazy(tiny), ord=-2))
    finch_assert_allclose(tiny_result, np.float64(1e-308))

    zero = np.array([0.0, 2.0], dtype=np.float64)
    zero_result = finch.compute(finch.linalg.vector_norm(finch.lazy(zero), ord=-2))
    finch_assert_allclose(zero_result, np.float64(0.0))

    nan = np.array([1.0, np.nan], dtype=np.float64)
    nan_result = finch.compute(finch.linalg.vector_norm(finch.lazy(nan), ord=-2))
    assert np.isnan(nan_result.item())


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"ord": 1},
        {"ord": -1},
        {"ord": float("inf")},
        {"ord": -float("inf"), "keepdims": True},
        {"ord": 2},
        {"ord": -2},
        {"ord": "nuc"},
    ],
)
@pytest.mark.parametrize(
    "wrap",
    [
        lambda x: x,
        TestOverrideTensor,
    ],
)
def test_linalg_matrix_norm_eager(kw, wrap):
    x = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)
    wx = wrap(x)
    expected = np.linalg.matrix_norm(x, **kw)

    result = finch.linalg.matrix_norm(wx, **kw)

    assert finch.ftype(expected.dtype.type) == result.element_type
    finch_assert_allclose(result, expected)


@pytest.mark.usefixtures("interpreter_scheduler")
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"ord": 1},
        {"ord": -1},
        {"ord": float("inf")},
        {"ord": -float("inf"), "keepdims": True},
    ],
)
def test_linalg_matrix_norm_lazy(kw):
    x = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float64)
    expected = np.linalg.matrix_norm(x, **kw)

    result = finch.linalg.matrix_norm(finch.lazy(x), **kw)
    result = finch.compute(result)

    assert finch.ftype(expected.dtype.type) == result.element_type
    finch_assert_allclose(result, expected)


def test_linalg_matrix_norm_lazy_eager_only_warns_and_computes():
    x = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float64)
    expected = np.linalg.matrix_norm(x, ord=2)

    with pytest.warns(RuntimeWarning, match="matrix_norm"):
        result = finch.linalg.matrix_norm(finch.lazy(x), ord=2)

    assert not isinstance(result, finch.LazyTensor)
    finch_assert_allclose(result, expected)


def test_linalg_inv_lazy_warns_and_computes():
    x = np.array([[1.0, 2.0], [3.0, 5.0]])
    expected = np.linalg.inv(x)

    with pytest.warns(RuntimeWarning, match="inv"):
        result = finch.linalg.inv(finch.lazy(x))

    assert not isinstance(result, finch.LazyTensor)
    finch_assert_allclose(result, expected)


@pytest.mark.parametrize(
    "name",
    [
        "cholesky",
        "cross",
        "det",
        "eigh",
        "eigvalsh",
        "matrix_rank",
        "lu",
        "pinv",
        "qr",
        "slogdet",
        "solve",
        "svd",
        "svdvals",
    ],
)
def test_linalg_missing_methods_are_exposed(name):
    assert hasattr(finch.linalg, name)


@pytest.mark.parametrize(
    "name",
    [
        "fft",
        "ifft",
        "fftn",
        "ifftn",
        "rfft",
        "irfft",
        "rfftn",
        "irfftn",
        "hfft",
        "ihfft",
        "fftshift",
        "ifftshift",
        "fftfreq",
        "rfftfreq",
    ],
)
def test_fft_methods_are_exposed(name):
    assert hasattr(finch.fft, name)


def test_linalg_new_eager_methods_use_numpy_fallback():
    x = np.array([[3.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    finch_assert_allclose(finch.linalg.det(x), np.linalg.det(x))
    finch_assert_allclose(finch.linalg.solve(x, b), np.linalg.solve(x, b))
    finch_assert_allclose(
        finch.linalg.svdvals(x),
        np.linalg.svd(x, compute_uv=False),
    )


def test_linalg_cross_uses_lazy_formula():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    result = finch.linalg.cross(finch.lazy(x), finch.lazy(y))

    assert isinstance(result, finch.LazyTensor)
    finch_assert_allclose(finch.compute(result), np.cross(x, y))


def test_linalg_cross_supports_axis_lazy_formula():
    x = np.arange(6.0).reshape(3, 2)
    y = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    result = finch.linalg.cross(finch.lazy(x), y, axis=0)

    assert isinstance(result, finch.LazyTensor)
    finch_assert_allclose(finch.compute(result), np.cross(x, y, axis=0))


def test_linalg_sparse_det_uses_superlu():
    x = scipy_sparse.csc_matrix(
        np.array(
            [
                [0.0, 2.0, 0.0],
                [3.0, 0.0, 4.0],
                [0.0, 5.0, 6.0],
            ]
        )
    )

    result = finch.linalg.det(x)

    finch_assert_allclose(result, np.linalg.det(x.toarray()))


def test_linalg_lu_uses_dense_fallback():
    x = np.array([[2.0, 5.0, 8.0], [5.0, 2.0, 2.0], [7.0, 5.0, 6.0]])

    p, lower, upper = finch.linalg.lu(x)

    finch_assert_allclose(p @ lower @ upper, x)


def test_linalg_lu_uses_sparse_superlu():
    x = scipy_sparse.csc_matrix(
        np.array([[4.0, 0.0, 1.0], [0.0, 3.0, 2.0], [1.0, 0.0, 5.0]])
    )
    b = np.array([1.0, 2.0, 3.0])

    result = finch.linalg.lu(x)

    finch_assert_allclose(result.solve(b), np.linalg.solve(x.toarray(), b))


def test_linalg_partial_sparse_eigen_kwargs():
    x = scipy_sparse.diags([1.0, 2.0, 3.0, 4.0], format="csr")

    vals = finch.linalg.eigvalsh(x, k=2, rtol=1e-12)
    eig_vals, eig_vecs = finch.linalg.eigh(x, k=2, atol=1e-12)

    finch_assert_allclose(np.sort(vals.to_numpy()), np.array([3.0, 4.0]))
    finch_assert_allclose(np.sort(eig_vals.to_numpy()), np.array([3.0, 4.0]))
    assert eig_vecs.shape == (4, 2)


def test_linalg_partial_sparse_svd_kwargs():
    x = scipy_sparse.diags([1.0, 2.0, 3.0, 4.0], format="csr")

    vals = finch.linalg.svdvals(x, k=2, rtol=1e-12)
    u, s, vh = finch.linalg.svd(x, k=2, atol=1e-12)

    finch_assert_allclose(np.sort(vals.to_numpy()), np.array([3.0, 4.0]))
    finch_assert_allclose(np.sort(s.to_numpy()), np.array([3.0, 4.0]))
    assert u.shape == (4, 2)
    assert vh.shape == (2, 4)


def test_linalg_partial_sparse_kwargs_dense_fallback_returns_full_results():
    x = np.diag([1.0, 2.0, 3.0, 4.0])

    with pytest.warns(RuntimeWarning, match="eigvalsh dense fallback"):
        eig_vals = finch.linalg.eigvalsh(x, k=2, rtol=1e-12)
    with pytest.warns(RuntimeWarning, match="svdvals dense fallback"):
        singular_vals = finch.linalg.svdvals(x, k=2, atol=1e-12)

    finch_assert_allclose(eig_vals, np.linalg.eigvalsh(x))
    finch_assert_allclose(singular_vals, np.linalg.svd(x, compute_uv=False))


def test_linalg_partial_sparse_warns_when_combining_tolerances():
    x = scipy_sparse.diags([1.0, 2.0, 3.0, 4.0], format="csr")

    with pytest.warns(RuntimeWarning, match="eigvalsh sparse fallback"):
        eig_vals = finch.linalg.eigvalsh(x, k=2, rtol=1e-12, atol=1e-12)
    with pytest.warns(RuntimeWarning, match="svdvals sparse fallback"):
        singular_vals = finch.linalg.svdvals(x, k=2, rtol=1e-12, atol=1e-12)

    finch_assert_allclose(np.sort(eig_vals.to_numpy()), np.array([3.0, 4.0]))
    finch_assert_allclose(np.sort(singular_vals.to_numpy()), np.array([3.0, 4.0]))


def test_linalg_matrix_rank_accepts_atol():
    x = np.diag([1.0, 1e-12])

    result = finch.linalg.matrix_rank(x, atol=1e-10)

    finch_assert_equal(result, np.linalg.matrix_rank(x, tol=1e-10))


def test_linalg_matrix_rank_warns_when_combining_tolerances():
    x = np.diag([1.0, 1e-12])

    with pytest.warns(RuntimeWarning, match="matrix_rank cannot apply both"):
        result = finch.linalg.matrix_rank(x, rtol=1e-12, atol=1e-10)

    finch_assert_equal(result, np.linalg.matrix_rank(x, tol=1e-10))


def test_fft_eager_methods_use_numpy_fallback():
    x = np.arange(4, dtype=np.float64)

    finch_assert_allclose(finch.fft.fft(x), np.fft.fft(x))
    finch_assert_allclose(finch.fft.rfft(x), np.fft.rfft(x))
    finch_assert_allclose(finch.fft.fftfreq(4), np.fft.fftfreq(4))
    assert finch.fft.fftfreq(4, dtype=finch.float32).to_numpy().dtype == np.float32
    assert finch.fft.rfftfreq(4, dtype=finch.float32).to_numpy().dtype == np.float32


def test_new_eager_only_methods_warn_compute_lazy_operands():
    x = np.array([[3.0, 1.0], [1.0, 3.0]])

    with pytest.warns(RuntimeWarning, match="det"):
        det_result = finch.linalg.det(finch.lazy(x))
    with pytest.warns(RuntimeWarning, match="fft"):
        fft_result = finch.fft.fft(finch.lazy(x[0]))

    finch_assert_allclose(det_result, np.linalg.det(x))
    finch_assert_allclose(fft_result, np.fft.fft(x[0]))


def test_new_lazy_methods_error_directly():
    lazy_mod = importlib.import_module("finch.interface.lazy")
    x = finch.lazy(np.eye(2))

    with pytest.raises(NotImplementedError, match="det is eager-only"):
        lazy_mod.det(x)
    with pytest.raises(NotImplementedError, match="fft is eager-only"):
        lazy_mod.fft(x)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "x, axis, expected",
    [
        (np.array([[1], [2]]), 1, np.array([1, 2])),
        (np.array([[[3]]]), (0, 1), np.array([3])),
        (np.zeros((1, 2, 1, 3)), (0, 2), np.zeros((2, 3))),
        (np.array([[[1, 2, 3]]]), 0, np.array([[1, 2, 3]])),
    ],
)
def test_squeeze_valid(x, axis, expected):
    """
    Tests for squeeze operation
    """
    result = finch.squeeze(x, axis=axis)
    finch_assert_equal(result, expected)


@pytest.mark.parametrize(
    "x, axis",
    [
        (np.array([[1, 2], [3, 4]]), 0),  # axis 0 is not singleton
        (np.zeros((2, 1, 3)), (0, 2)),  # axis 0 and 2 not both singleton
        (np.ones((1, 2, 1)), (1,)),  # axis 1 is not singleton
    ],
)
def test_squeeze_invalid(x, axis):
    with pytest.raises(ValueError):
        finch.squeeze(x, axis=axis)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "x, axis",
    [
        (np.array([1, 2, 3]), 0),
        (np.array([1, 2, 3]), 1),
        (np.array([[1, 2], [3, 4]]), 0),
        (np.array([[1, 2], [3, 4]]), 1),
        (np.array([[1, 2], [3, 4]]), 2),
        (np.array([1, 2, 3]), -1),
        (np.array([1, 2, 3]), -2),
        (np.array([[1, 2], [3, 4]]), -1),
        (np.array([[1, 2], [3, 4]]), -3),
    ],
)
def test_expand_dims_valid(x, axis):
    expected = np.expand_dims(x, axis=axis)
    result = finch.expand_dims(x, axis=axis)
    finch_assert_equal(result, expected)


@pytest.mark.parametrize(
    "x, axis",
    [
        (np.array([1, 2, 3]), 3),  # out of bounds
        (np.array([1, 2, 3]), -4),  # out of bounds
        (np.array([[1, 2], [3, 4]]), 4),
        (np.array([[1, 2], [3, 4]]), -4),
    ],
)
def test_expand_dims_invalid(x, axis):
    with pytest.raises(IndexError):
        finch.expand_dims(x, axis=axis)


@pytest.mark.parametrize(
    "x",
    [
        0,
        0.0,
        -4,
        1,
        2.4,
        -1.54,
        True,
        False,
        float("inf"),
        float("-inf"),
        float("nan"),
        complex(1, 2),
        complex(0, 0),
    ],
)
@pytest.mark.parametrize("func", [complex, int, float, bool])
def test_scalar_coerce(x, func):
    """
    Tests for scalar coercion to different types.
    """
    if isinstance(x, complex) and func in [int, float]:
        # no defined behavior in spec
        return

    try:
        expected = func(x)
    except (ValueError, TypeError, OverflowError):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            print(func(TestOverrideTensor(np.array(x))))
        return
    result = func(TestOverrideTensor(np.array(x)))
    assert isinstance(result, func), f"Result should be of type {func.__name__}"
    works = result == expected or np.isnan(result) and np.isnan(expected)
    assert works, f"Expected {expected}, got {result}"


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "arrays, axis",
    [
        (
            (
                np.arange(6, dtype=np.int64).reshape(2, 3),
                np.arange(6, 15, dtype=np.int64).reshape(3, 3),
            ),
            0,
        ),
        (
            (
                np.arange(6, dtype=np.int64).reshape(2, 3),
                np.arange(6, 10, dtype=np.int64).reshape(2, 2),
            ),
            1,
        ),
        (
            (
                np.arange(24, dtype=np.int64).reshape(2, 3, 4),
                np.arange(24, 48, dtype=np.int64).reshape(2, 3, 4),
            ),
            -1,
        ),
    ],
)
@pytest.mark.parametrize(
    "array_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
def test_concat(arrays, axis, array_wrap):
    wrapped = tuple(array_wrap(array) for array in arrays)
    result = finch.concat(wrapped, axis=axis)
    expected = np.concatenate(arrays, axis=axis)

    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)
    finch_assert_equal(result, expected)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "array_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
    ],
)
def test_concat_axis_none_eager(array_wrap):
    arrays = (
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.arange(6, 8, dtype=np.int64).reshape(1, 2),
    )
    wrapped = tuple(array_wrap(array) for array in arrays)

    result = finch.concat(wrapped, axis=None)

    finch_assert_equal(
        result,
        np.concatenate(tuple(array.reshape(-1) for array in arrays), axis=0),
    )


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
def test_concat_axis_none_lazy():
    raw_arrays = (
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.arange(6, 8, dtype=np.int64).reshape(1, 2),
    )
    arrays = tuple(finch.lazy(array) for array in raw_arrays)

    result = finch.compute(finch.concat(arrays, axis=None))

    finch_assert_equal(
        result,
        np.concatenate(tuple(array.reshape(-1) for array in raw_arrays), axis=0),
    )


def test_concat_axis_none_promotes_dtype():
    arrays = (
        finch.asarray(np.array([1], dtype=np.int8)),
        finch.asarray(np.array([128], dtype=np.int16)),
    )

    result = finch.concat(arrays, axis=None)
    expected = np.concatenate(tuple(array.to_numpy().reshape(-1) for array in arrays))

    assert result.dtype == ftype(expected.dtype.type)
    finch_assert_equal(result, expected)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
def test_concat_uses_first_fill_value():
    a = finch.BufferizedNDArray.from_numpy(
        np.array([[0, 1], [2, 0]], dtype=np.int64),
        fill_value=0,
    )
    b = finch.BufferizedNDArray.from_numpy(
        np.array([[7, 3]], dtype=np.int64),
        fill_value=7,
    )

    result = finch.concat((finch.lazy(a), finch.lazy(b)), axis=0)

    assert result.fill_value == a.fill_value
    result = finch.compute(result)
    assert result.fill_value == a.fill_value
    finch_assert_equal(result, np.concatenate((a.to_numpy(), b.to_numpy()), axis=0))


def test_concat_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="dimensions except"):
        finch.concat((np.ones((2, 3)), np.ones((3, 4))), axis=0)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
def test_lazy_shape_ops_use_symbolic_selectors():
    array = np.arange(24, dtype=np.int64).reshape(2, 3, 4)

    cases = [
        (
            finch.flip(finch.lazy(array), axis=(0, 2)),
            np.flip(array, axis=(0, 2)),
        ),
        (
            finch.roll(finch.lazy(array), shift=(1, -2), axis=(0, 2)),
            np.roll(array, shift=(1, -2), axis=(0, 2)),
        ),
        (
            finch.take(finch.lazy(array), finch.asarray([2, 0]), axis=1),
            np.take(array, [2, 0], axis=1),
        ),
        (
            finch.repeat(finch.lazy(array), 2, axis=None),
            np.repeat(array, 2, axis=None),
        ),
        (
            finch.tile(finch.lazy(array), (2, 1, 1)),
            np.tile(array, (2, 1, 1)),
        ),
    ]

    for result, expected in cases:
        assert isinstance(result, finch.LazyTensor)
        finch_assert_equal(finch.compute(result), expected)


def test_repeat_rejects_data_dependent_repeats():
    with pytest.raises(NotImplementedError, match="data-dependent output shape"):
        finch.repeat(
            finch.lazy(np.arange(6, dtype=np.int64).reshape(2, 3)),
            finch.asarray([1, 2, 1]),
            axis=1,
        )


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
def test_lazy_stack_and_unstack():
    arrays = (
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.arange(6, 12, dtype=np.int64).reshape(2, 3),
    )

    stacked = finch.stack(tuple(finch.lazy(array) for array in arrays), axis=1)

    assert isinstance(stacked, finch.LazyTensor)
    finch_assert_equal(finch.compute(stacked), np.stack(arrays, axis=1))

    parts = finch.unstack(stacked, axis=1)
    assert isinstance(parts, tuple)
    assert len(parts) == len(arrays)
    for part, expected in zip(parts, arrays, strict=True):
        assert isinstance(part, finch.LazyTensor)
        finch_assert_equal(finch.compute(part), expected)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
def test_lazy_take_along_axis():
    array = np.arange(6, dtype=np.int64).reshape(2, 3)
    indices = np.array([[2, 0], [1, 1]], dtype=np.intp)

    result = finch.take_along_axis(finch.lazy(array), indices, axis=1)

    assert isinstance(result, finch.LazyTensor)
    finch_assert_equal(
        finch.compute(result), np.take_along_axis(array, indices, axis=1)
    )


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "array, shape",
    [
        (np.arange(6, dtype=np.int64).reshape(2, 3), (3, 2)),
        (np.arange(6, dtype=np.int64).reshape(2, 3), (-1,)),
        (np.arange(6, dtype=np.int64), (2, 3)),
        (np.array(5, dtype=np.int64), (1,)),
    ],
)
def test_lazy_reshape(array, shape):
    result = finch.reshape(finch.lazy(array), shape)

    assert isinstance(result, finch.LazyTensor)
    finch_assert_equal(finch.compute(result), np.reshape(array, shape))


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
def test_lazy_reshape_preserves_fill_value():
    array = finch.BufferizedNDArray.from_numpy(
        np.array([[9, 1], [2, 9]], dtype=np.int64),
        fill_value=9,
    )

    result = finch.reshape(finch.lazy(array), (4,))

    assert result.fill_value == array.fill_value
    result = finch.compute(result)
    finch_assert_equal(result, array.to_numpy().reshape((4,)))


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
def test_lazy_reshape_uses_single_mask(monkeypatch):
    lazy_module = importlib.import_module("finch.interface.lazy")
    original = lazy_module.ReshapeMaskTensor
    mask_shapes = []

    def recording_mask(old_shape, new_shape, *args, **kwargs):
        mask_shapes.append((tuple(old_shape), tuple(new_shape)))
        return original(old_shape, new_shape, *args, **kwargs)

    monkeypatch.setattr(lazy_module, "ReshapeMaskTensor", recording_mask)
    array = np.arange(24, dtype=np.int64).reshape(2, 3, 4)

    result = finch.reshape(finch.lazy(array), (6, 4))

    assert mask_shapes == [((2, 3, 4), (6, 4))]
    finch_assert_equal(finch.compute(result), array.reshape(6, 4))


def test_lazy_reshape_rejects_invalid_shape():
    with pytest.raises(ValueError, match="Cannot reshape"):
        finch.reshape(finch.lazy(np.arange(6)), (4,))


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "x, shape",
    [
        # ——— VALID CASES ———
        # scalar → 2×3
        (np.array(5), (2, 3)),
        # 1D int → 2×3
        (np.array([1, 2, 3]), (2, 3)),
        # 1D float → 1×3×2 (prepend one axis)
        (np.array([0.5, 1.5]), (1, 2, 2)),
        # 2D bool → 2×2×3 (prepend one axis)
        (np.array([[True, False, True]]), (2, 1, 3)),
        # 1-element → 2×1×3
        (np.array([7.0 + 4.2j]), (2, 1, 3)),
        # already matching shape (no change)
        (np.arange(6).reshape(2, 3), (2, 3)),
        # zero-length axis: (0,) → (4, 0)
        (np.empty((0,)), (4, 0)),
        # broadcast in middle: (1,4 +1j,1) → (3,4,2)
        (np.ones((1, 4, 1)), (3, 4, 2)),
        (np.arange(4).reshape(2, 2), (2, 2, 2)),
        # 1-dim can be broadcast to 0-dim: (3,1) → (3, 0)
        (np.array([1, 2, 3]).reshape(-1, 1), (3, 0)),
        # 0-dim can be prepended: (3, 2) → (0, 3, 2)
        (np.arange(6).reshape(3, 2), (0, 3, 2)),
        # ——— INVALID CASES ———
        # mismatched non-1 dim at end
        (np.array([1, 2, 3]), (2, 2)),
        # mismatched non-1 dim in middle
        (np.ones((2, 3, 4)), (2, 5, 4)),
        # broadcast on right side
        (np.arange(3).reshape(3, 1), (2, 3, 1, 5)),
    ],
)
@pytest.mark.parametrize(
    "x_wrap",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
def test_broadcast_to(x, shape, x_wrap):
    """
    Tests for broadcasting an array to a specified shape.
    """
    wx = x_wrap(x)
    # try NumPy’s broadcast_to first
    try:
        expected = np.broadcast_to(x, shape)
    except ValueError:
        # if NumPy cannot broadcast, we expect finch to raise
        with pytest.raises(ValueError):
            finch.broadcast_to(wx, shape)

    else:
        out = finch.broadcast_to(wx, shape)
        if isinstance(wx, finch.LazyTensor):
            out = finch.compute(out)
        finch_assert_equal(out, expected, strict=True)


@pytest.mark.parametrize("indexing", ["xy", "ij"])
def test_meshgrid(indexing):
    arrays = [
        np.array([1, 2], dtype=np.int32),
        np.array([3, 4, 5], dtype=np.int32),
        np.array([6, 7], dtype=np.int32),
    ]
    expected = np.meshgrid(*arrays, indexing=indexing)
    result = finch.meshgrid(*arrays, indexing=indexing)
    lazy_result = finch.meshgrid(
        *(finch.lazy(array) for array in arrays),
        indexing=indexing,
    )

    for result_arr, lazy_arr, expected_arr in zip(
        result, lazy_result, expected, strict=True
    ):
        finch_assert_equal(result_arr, expected_arr, strict=True)
        finch_assert_equal(finch.compute(lazy_arr), expected_arr, strict=True)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "shapes",
    [
        ((1, 2, 3), (4, 2, 3), (3,), (9, 4, 2, 3), (9, 1, 1, 3)),
        ((7, 2, 3, 4), (2, 3, 1), (2, 1, 1), (1, 2, 1, 1), (1,)),
        ((1,), (1,)),
        ((2, 3), (3, 2)),  # error
        ((1,), (4, 0)),
        ((0,), (1, 0)),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
def test_broadcast_arrays(shapes, wrapper, rng, random_wrapper):
    """
    Tests for broadcasting multiple arrays to a common shape.
    The wrapper is randomly applied to each shape to ensure
    """

    # Generate random arrays for each shape
    arrays = [rng.random(shape) for shape in shapes]
    wrapped_arrays = random_wrapper(arrays, wrapper)
    try:
        expected = np.broadcast_arrays(*arrays)
    except ValueError:
        with pytest.raises(ValueError):
            finch.broadcast_arrays(*wrapped_arrays)
        return
    result = finch.broadcast_arrays(*wrapped_arrays)
    if isinstance(result[0], finch.LazyTensor):
        assert all(isinstance(r, finch.LazyTensor) for r in result)
        result = finch.compute(result)  # compute all lazy tensors

    assert len(result) == len(expected), "Number of results does not match expected"
    for i, (res, exp) in enumerate(zip(result, expected, strict=True)):
        assert res.shape == exp.shape, (
            f"Shape mismatch: got {res.shape}, expected {exp.shape} at index {i}"
        )
        finch_assert_equal(res, exp)


@pytest.mark.usefixtures("interpreter_scheduler")  # TODO: remove
@pytest.mark.parametrize(
    "shape, source, destination",
    [
        ((3, 4, 5), 0, -3),
        ((21, 1, 3, 2, 0), -1, -2),
        ((2, 3, 4), (0, 1), (2, 0)),
        ((5, 4, 3), (0, 1, 2), (-1, -2, -3)),
        ((5, 8, 9, 4, 3), (0, 1), (-1, 4)),  # error case
        ((5, 8, 9, 4, 3), (-9, 1), (-1, 74)),  # error case
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        lambda x: x,
        TestOverrideTensor,
        finch.lazy,
    ],
)
def test_moveaxis(shape, source, destination, wrapper, rng):
    """
    Tests for moving axes of an array to a new position.
    """
    # Generate a random array with the specified shape
    x = rng.random(shape)
    # Apply wrapper to input
    wrapped_x = wrapper(x)
    # Compute expected result using NumPy
    try:
        expected = np.moveaxis(x, source, destination)
    except ValueError:
        # If NumPy raises a ValueError, we expect finch to raise the same
        with pytest.raises(ValueError):
            finch.moveaxis(wrapped_x, source, destination)
        return

    result = finch.moveaxis(wrapped_x, source, destination)
    if isinstance(result, finch.LazyTensor):
        result = finch.compute(result)

    finch_assert_equal(result, expected, strict=True)


@pytest.mark.skip(
    "We're holding off on numba tests for tril until we can refactor looplets "
    "and build a full suite of mask tensors."
)
@pytest.mark.usefixtures("numba_compiler")
@pytest.mark.parametrize(
    "arr1,arr2",
    [
        (
            np.array([[2, 0, 3], [1, 3, -3], [6, 0, 1]]),
            np.array([[-4, 2, 1], [0, 0, -3], [4, 9, 11]]),
        ),
        (
            np.full((5, 8), 9, dtype=np.float64),
            np.ones((5, 8), dtype=np.float64),
        ),
        (
            np.full((3, 4, 3), 4, dtype=np.int64),
            np.full((3, 4, 3), 3, dtype=np.int64),
        ),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        lambda x: x,
        finch.lazy,
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        lambda xp, x, y: xp.multiply(x, y),
        lambda xp, x, y: xp.add(x, y),
        lambda xp, x, _: xp.sum(x, axis=0),
    ],
)
def test_tril(arr1: np.ndarray, arr2: np.ndarray, wrapper, op):
    # construct dense format
    fmt = finch.element(
        arr1.dtype.type(0),
        ftype(arr1.dtype),
        ftype(np.intp),
        finch.NumpyBufferFType,
    )
    for _ in range(arr1.ndim):
        fmt = finch.dense(fmt)
    fmt = finch.fiber_tensor(fmt)

    f_arr = finch.asarray(arr1, format=fmt)
    tril_arr = finch.tril(f_arr)
    f_arr_2 = finch.asarray(arr2, format=fmt)

    wrap_arr = wrapper(tril_arr)
    wrap_arr_2 = wrapper(f_arr_2)
    plan = op(finch, wrap_arr, wrap_arr_2)
    result = finch.compute(plan) if isinstance(plan, finch.LazyTensor) else plan

    expected = op(np, np.tril(arr1), arr2)
    finch_assert_equal(result, expected, strict=True)


def test_eager_compute():
    # Test that compute on an eager tensor returns the same tensor
    x = np.array([[1, 2], [3, 4]])
    eager_tensor = TestOverrideTensor(x)
    lazy_tensor = finch.add(
        finch.lazy(eager_tensor), finch.lazy(eager_tensor)
    )  # This should return an eager tensor
    eager_result, lazy_result = finch.compute((eager_tensor, lazy_tensor))
    assert eager_result is eager_tensor, (
        "Compute on eager tensor should return the same tensor"
    )
    finch_assert_equal(eager_result, x)
    finch_assert_equal(lazy_result, (2 * eager_tensor))
