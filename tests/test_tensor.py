import pytest

import numpy as np
import scipy.sparse

import finchlite
from finchlite import (
    NumpyBuffer,
    NumpyBufferFType,
    asarray,
    dense,
    element,
    fiber_tensor,
    sparse_list,
)
from finchlite.tensor import (
    BufferizedNDArray,
    EyeTensor,
    FillTensor,
    IndexTensor,
    LowerTriangleTensor,
    OddEvenMergeSortLowerMaskTensor,
    OddEvenMergeSortPartnerMaskTensor,
    OneHotMaskTensor,
    PairCarryTensor,
    PairSumTensor,
    ParityMaskTensor,
    PatternTensor,
    RepeatTensor,
    ReshapeMaskTensor,
    ReverseTensor,
    RollTensor,
    UpperTriangleTensor,
)
from finchlite.tensor.traits import (
    Blocked as BlockedProperty,
)
from finchlite.tensor.traits import (
    Dense as DenseProperty,
)
from finchlite.tensor.traits import (
    Repeated as RepeatedProperty,
)


def test_fiber_tensor_attributes():
    fmt = fiber_tensor(dense(dense(element(0.0, finchlite.float64))))
    shape = (3, 4)
    arr = np.ones(shape)
    a = asarray(arr, format=fmt)

    # Check shape attribute
    assert a.shape == shape

    # Check ndim
    assert a.ndim == 2

    # Check shape_type
    assert a.shape_type == (finchlite.intp, finchlite.intp)

    # Check element_type
    assert a.element_type == finchlite.float64

    # Check fill_value
    assert a.fill_value == 0

    # Check position_type
    assert a.position_type == finchlite.intp

    # Check buffer_format exists
    assert a.buffer_factory == NumpyBufferFType


def test_bufferized_ndarray_fill_value_dtype():
    for dtype in (np.int32, np.float32, np.bool_):
        arr = np.ones((2, 3), dtype=dtype)
        a = BufferizedNDArray.from_numpy(arr, fill_value=0)

        assert a.fill_value.dtype == arr.dtype
        assert a.ftype.fill_value.dtype == arr.dtype
        assert finchlite.lazy(a).fill_value.dtype == arr.dtype
        assert asarray(a, copy=True).fill_value.dtype == arr.dtype
        assert a[0:1].fill_value.dtype == arr.dtype
        assert a.reshape((3, 2)).fill_value.dtype == arr.dtype

    full = finchlite.full((2, 3), 7, dtype=np.int32)
    assert full.fill_value.dtype == np.dtype(np.int32)


def test_bufferized_ndarray_custom_fill_value():
    arr = np.ones((2, 3), dtype=np.float32)
    a = BufferizedNDArray.from_numpy(arr, fill_value=2)

    assert a.fill_value == np.float32(2)
    assert a.ftype.fill_value == np.float32(2)
    assert np.all(a.ftype.construct((2, 3)).to_numpy() == np.float32(2))

    x = BufferizedNDArray.from_numpy(arr, fill_value=np.nan)
    y = BufferizedNDArray.from_numpy(arr, fill_value=np.nan)
    assert finchlite.same(x.fill_value, y.fill_value)
    assert x.ftype == y.ftype
    assert hash(x.ftype) == hash(y.ftype)


def test_bufferized_ndarray_to_numpy_returns_view():
    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    tensor = BufferizedNDArray.from_numpy(arr)

    result = tensor.to_numpy()

    np.testing.assert_array_equal(result, arr)
    assert np.shares_memory(result, tensor.val.arr)


def test_numpy_buffer_tuple_dtype_load_store():
    dtype = np.dtype([("element_0", np.float64), ("element_1", np.int32)])
    buf = NumpyBuffer(np.zeros(2, dtype=dtype))

    assert buf.ftype.element_type == finchlite.algebra.TupleFType.from_tuple(
        (finchlite.float64, finchlite.int32)
    )
    assert buf.load(0) == (np.float64(0.0), np.int32(0))

    buf.store(1, (np.float64(1.5), np.int32(2)))

    assert buf.load(1) == (np.float64(1.5), np.int32(2))


def test_tensor_conversion_helpers_accept_scipy_sparse():
    arr = np.array([[0, 2, 0], [3, 0, 4]], dtype=np.int32)
    scipy_tensor = scipy.sparse.csr_matrix(arr)

    np.testing.assert_array_equal(finchlite.to_numpy(scipy_tensor), arr)
    assert finchlite.to_scipy(scipy_tensor) is scipy_tensor


def test_dense_tensor_to_scipy_rejects():
    tensor = BufferizedNDArray.from_numpy(np.arange(3, dtype=np.int32))

    with pytest.raises(NotImplementedError, match="does not support to_scipy"):
        tensor.to_scipy()


def test_empty_like_preserves_fill_value():
    import importlib

    lazy_interface = importlib.import_module("finchlite.interface.lazy")
    arr = np.ones((2, 3), dtype=np.int32)
    x = BufferizedNDArray.from_numpy(arr, fill_value=5)

    lazy_x = finchlite.lazy(x)
    lazy_out = lazy_interface.empty_like(lazy_x)
    out = finchlite.empty_like(x)

    assert lazy_out.fill_value == np.int32(5)
    assert lazy_out.element_type == finchlite.int32
    assert out.fill_value == np.int32(5)
    assert out.element_type == finchlite.int32
    np.testing.assert_array_equal(out.to_numpy(), np.full((2, 3), 5, dtype=np.int32))


def test_index_tensor_returns_linear_indices():
    import importlib

    lazy_interface = importlib.import_module("finchlite.interface.lazy")

    tns = lazy_interface.IndexTensor((2, 3), np.int64)

    assert tns.shape == (2, 3)
    assert tns.element_type == finchlite.int64
    assert tns.fill_value == np.int64(0)
    assert tns[0, 0].item() == np.int64(0)
    assert tns[0, 2].item() == np.int64(2)
    assert tns[1, 0].item() == np.int64(3)
    assert tns[1, 2].item() == np.int64(5)

    constructed = tns.ftype.construct((2, 3))

    assert constructed.ftype == tns.ftype
    assert constructed[1, 2].item() == np.int64(5)


def test_masks_are_serial_but_operations_keep_input_device():
    cpu_dev = finchlite.cpu("mask-test", n=2)
    x = finchlite.asarray(np.eye(2), device=cpu_dev)

    assert EyeTensor((2, 2)).device == finchlite.serial()
    assert FillTensor((2, 2), 0).device == finchlite.serial()
    assert IndexTensor((2, 2)).device == finchlite.serial()
    assert ParityMaskTensor(2).device == finchlite.serial()
    assert finchlite.triu(x).device == cpu_dev
    y = finchlite.asarray(np.arange(3), device=cpu_dev)
    assert finchlite.diff(y).device == cpu_dev


def test_fiber_tensor():
    fmt = fiber_tensor(
        dense(
            dense(
                element(np.int64(0), finchlite.int64, finchlite.intp, NumpyBufferFType)
            )
        )
    )
    assert fmt.level_format_properties == [
        DenseProperty((), (0,)),
        DenseProperty((0,), (1,)),
    ]
    sparse_fmt = fiber_tensor(
        dense(
            sparse_list(
                dense(
                    element(
                        np.int64(0),
                        finchlite.int64,
                        finchlite.intp,
                        NumpyBufferFType,
                    )
                ),
                finchlite.intp,
            )
        )
    )
    assert sparse_fmt.level_format_properties == [
        DenseProperty((), (0,)),
        DenseProperty((0, 1), (2,)),
    ]

    asarray(np.arange(12).reshape((3, 4)), format=fmt)

    cpu_dev = finchlite.cpu("fiber-test", n=2)
    tensor = asarray(np.arange(12).reshape((3, 4)), format=fmt, device=cpu_dev)
    assert tensor.device == cpu_dev


def test_sparse_level_construct_initializes_empty_storage():
    elem_t = element(0.0, finchlite.float64, finchlite.intp, NumpyBufferFType)

    list_level = sparse_list(elem_t, finchlite.intp).construct((4,), pos=3)
    np.testing.assert_array_equal(list_level.ptr.arr, np.zeros(4, dtype=np.intp))
    assert list_level.idx.length() == 0
    assert list_level.lvl.val.length() == 0

    hash_level = finchlite.sparse_hash(elem_t, finchlite.intp).construct((4,), pos=3)
    np.testing.assert_array_equal(hash_level.ptr.arr, np.zeros(4, dtype=np.intp))
    assert hash_level.tbl_ctrl.length() == 0
    assert hash_level.tbl.length() == 0
    assert hash_level.pool.length() == 0
    assert hash_level.perm.length() == 0
    assert hash_level.lvl.val.length() == 0

    coo_level = finchlite.sparse_coo(elem_t, 2, finchlite.intp).construct((4, 5), pos=3)
    np.testing.assert_array_equal(coo_level.ptr.arr, np.zeros(4, dtype=np.intp))
    assert len(coo_level.tbl) == 2
    assert all(idx.length() == 0 for idx in coo_level.tbl)
    assert coo_level.lvl.val.length() == 0

    bytemap_level = finchlite.sparse_bytemap(elem_t, finchlite.intp).construct(
        (4,), pos=3
    )
    np.testing.assert_array_equal(bytemap_level.ptr.arr, np.zeros(4, dtype=np.intp))
    np.testing.assert_array_equal(bytemap_level.tbl.arr, np.zeros(12, dtype=np.bool_))
    assert bytemap_level.srt.length() == 0
    assert bytemap_level.lvl.val.length() == 12


def test_bufferized_ndarray_level_format_properties():
    tensor = BufferizedNDArray.from_numpy(np.zeros((2, 3), dtype=np.int32))

    assert tensor.ftype.level_format_properties == [
        DenseProperty((), (0,)),
        DenseProperty((0,), (1,)),
    ]


@pytest.mark.parametrize(
    "tensor, expected",
    [
        (
            FillTensor((2, 3), 0),
            [
                DenseProperty((), (0,)),
                DenseProperty((0,), (1,)),
                RepeatedProperty((), (0,)),
                RepeatedProperty((0,), (1,)),
            ],
        ),
        (
            IndexTensor((2, 3)),
            [DenseProperty((), (0,)), DenseProperty((0,), (1,))],
        ),
        (ReshapeMaskTensor((2, 3), (3, 2), dtype=np.bool_), []),
        (EyeTensor((2, 3)), []),
        (
            UpperTriangleTensor((3, 4)),
            [BlockedProperty((0,), (1,)), BlockedProperty((1,), (0,))],
        ),
        (
            LowerTriangleTensor((3, 4)),
            [BlockedProperty((0,), (1,)), BlockedProperty((1,), (0,))],
        ),
        (PairSumTensor((3, 6)), [BlockedProperty((0,), (1,))]),
        (PairCarryTensor((6, 3)), [BlockedProperty((1,), (0,))]),
        (ReverseTensor((3, 3)), []),
        (RollTensor((3, 3), k=1), []),
        (
            RepeatTensor((6, 3), k=2),
            [BlockedProperty((1,), (0,)), RepeatedProperty((1,), (0,))],
        ),
        (OddEvenMergeSortPartnerMaskTensor((8, 8), p=2, k=1), []),
        (OddEvenMergeSortLowerMaskTensor(8, p=2, k=1), []),
        (OneHotMaskTensor(5, index=2), []),
        (ParityMaskTensor(5), []),
    ],
)
def test_symbolic_tensor_level_format_properties(tensor, expected):
    assert tensor.ftype.level_format_properties == expected


@pytest.mark.parametrize(
    "make_tensor, expected",
    [
        (lambda: finchlite.eye(3, 4, dtype=np.int32), np.eye(3, 4, dtype=np.int32)),
        (
            lambda: finchlite.eye(3, 4, k=1, dtype=np.int32),
            np.eye(3, 4, k=1, dtype=np.int32),
        ),
    ],
)
def test_matrix_pattern_tensors(make_tensor, expected):
    tensor = make_tensor()
    actual = np.array(
        [
            [tensor[i, j].item() for j in range(tensor.shape[1])]
            for i in range(tensor.shape[0])
        ],
        dtype=expected.dtype,
    )

    assert tensor.shape == expected.shape
    assert tensor.fill_value.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_lazy_matrix_pattern_tensor_compute():
    tensor = finchlite.lazy(EyeTensor((2, 3), dtype=np.int32))
    result = finchlite.compute(tensor)

    assert tensor.shape == (2, 3)
    assert result.fill_value.dtype == np.dtype(np.int32)
    actual = np.array(
        [
            [result[i, j].item() for j in range(result.shape[1])]
            for i in range(result.shape[0])
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(actual, np.eye(2, 3, dtype=np.int32))


@pytest.mark.parametrize(
    "tensor",
    [
        EyeTensor((2, 3)),
        ParityMaskTensor(3),
        ReshapeMaskTensor((2, 3), (3, 2), dtype=np.bool_),
    ],
)
def test_pattern_tensors_share_parent(tensor):
    assert isinstance(tensor, PatternTensor)
    assert isinstance(tensor, finchlite.PatternTensor)
    assert isinstance(tensor.ftype.construct(tensor.shape), type(tensor))


@pytest.mark.parametrize(
    "tensor",
    [
        finchlite.ReshapeMaskTensor((2, 3), (3, 2), dtype=np.bool_),
        FillTensor((2, 3), 0),
        IndexTensor((2, 3)),
        ParityMaskTensor(3),
        EyeTensor((2, 3)),
    ],
)
def test_symbolic_tensors_reject_materialization(tensor):
    with pytest.raises(NotImplementedError, match="does not support to_numpy"):
        tensor.to_numpy()
    with pytest.raises(NotImplementedError, match="does not support to_scipy"):
        tensor.to_scipy()


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_triu_tril(k):
    arr = np.arange(12, dtype=np.int32).reshape((3, 4))

    np.testing.assert_array_equal(
        finchlite.triu(arr, k=k).to_numpy(), np.triu(arr, k=k)
    )
    np.testing.assert_array_equal(
        finchlite.tril(arr, k=k).to_numpy(), np.tril(arr, k=k)
    )


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_diag(k):
    vec = np.array([1, 2, 3], dtype=np.int32)
    mat = np.arange(12, dtype=np.int32).reshape((3, 4))

    np.testing.assert_array_equal(
        finchlite.diag(vec, k=k).to_numpy(), np.diag(vec, k=k)
    )
    np.testing.assert_array_equal(
        finchlite.diag(mat, k=k).to_numpy(), np.diag(mat, k=k)
    )


@pytest.mark.parametrize(
    "arr, axis, n",
    [
        (np.arange(5, dtype=np.int32), -1, 1),
        (np.array([0, 2, 7, 15], dtype=np.uint8), -1, 2),
        (np.arange(6, dtype=np.float32).reshape((2, 3)), -1, 1),
        (np.arange(24, dtype=np.int64).reshape((2, 3, 4)), 0, 1),
        (np.arange(24, dtype=np.int64).reshape((2, 3, 4)), 1, 2),
        (np.arange(24, dtype=np.int64).reshape((2, 3, 4)), 2, 2),
    ],
)
def test_diff(arr, axis, n):
    expected = np.diff(arr, axis=axis, n=n)

    np.testing.assert_array_equal(
        finchlite.diff(arr, axis=axis, n=n).to_numpy(),
        expected,
    )
    np.testing.assert_array_equal(
        finchlite.compute(
            finchlite.diff(finchlite.lazy(arr), axis=axis, n=n)
        ).to_numpy(),
        expected,
    )


def test_diff_n_zero():
    arr = np.arange(6, dtype=np.float32).reshape((2, 3))

    np.testing.assert_array_equal(finchlite.diff(arr, n=0).to_numpy(), arr)
    np.testing.assert_array_equal(
        finchlite.compute(finchlite.diff(finchlite.lazy(arr), n=0)).to_numpy(),
        arr,
    )


def test_diff_empty_axis():
    arr = np.arange(3, dtype=np.int32)
    expected = np.diff(arr, n=4)

    np.testing.assert_array_equal(finchlite.diff(arr, n=4).to_numpy(), expected)
    np.testing.assert_array_equal(
        finchlite.compute(finchlite.diff(finchlite.lazy(arr), n=4)).to_numpy(),
        expected,
    )


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.uint8, np.uint16])
def test_sum_integer_accumulation_dtype(dtype):
    arr = np.arange(6, dtype=dtype)
    result = finchlite.sum(arr)

    assert result.to_numpy().dtype == np.sum(arr).dtype


@pytest.mark.parametrize(
    "arr, axis, dtype, include_initial",
    [
        (np.arange(5, dtype=np.int32), None, None, False),
        (np.arange(6, dtype=np.float32).reshape((2, 3)), 1, None, False),
        (np.arange(6, dtype=np.int16).reshape((2, 3)), 0, np.int64, False),
        (np.arange(6, dtype=np.uint8).reshape((2, 3)), -1, None, True),
        (np.arange(6, dtype=np.float64).reshape((2, 3)), 1, np.float32, True),
    ],
)
def test_cumulative_sum(arr, axis, dtype, include_initial):
    np_axis = 0 if axis is None else axis
    expected = np.cumsum(arr, axis=np_axis, dtype=dtype)
    if include_initial:
        pad_shape = list(expected.shape)
        pad_shape[np_axis] = 1
        expected = np.concatenate(
            (np.zeros(tuple(pad_shape), dtype=expected.dtype), expected),
            axis=np_axis,
        )

    result = finchlite.cumulative_sum(
        arr, axis=axis, dtype=dtype, include_initial=include_initial
    )
    lazy_result = finchlite.compute(
        finchlite.cumulative_sum(
            finchlite.lazy(arr),
            axis=axis,
            dtype=dtype,
            include_initial=include_initial,
        )
    )

    np.testing.assert_array_equal(result.to_numpy(), expected)
    np.testing.assert_array_equal(lazy_result.to_numpy(), expected)
    assert result.to_numpy().dtype == expected.dtype
    assert lazy_result.to_numpy().dtype == expected.dtype


@pytest.mark.parametrize(
    "arr, axis, dtype, include_initial",
    [
        (np.arange(1, 6, dtype=np.int32), None, None, False),
        (np.arange(1, 7, dtype=np.float32).reshape((2, 3)), 1, None, False),
        (np.arange(1, 7, dtype=np.int16).reshape((2, 3)), 0, np.int64, False),
        (np.arange(1, 7, dtype=np.uint8).reshape((2, 3)), -1, None, True),
        (np.arange(1, 7, dtype=np.float64).reshape((2, 3)), 1, np.float32, True),
    ],
)
def test_cumulative_prod(arr, axis, dtype, include_initial):
    np_axis = 0 if axis is None else axis
    expected = np.cumprod(arr, axis=np_axis, dtype=dtype)
    if include_initial:
        pad_shape = list(expected.shape)
        pad_shape[np_axis] = 1
        expected = np.concatenate(
            (np.ones(tuple(pad_shape), dtype=expected.dtype), expected),
            axis=np_axis,
        )

    result = finchlite.cumulative_prod(
        arr, axis=axis, dtype=dtype, include_initial=include_initial
    )
    lazy_result = finchlite.compute(
        finchlite.cumulative_prod(
            finchlite.lazy(arr),
            axis=axis,
            dtype=dtype,
            include_initial=include_initial,
        )
    )

    np.testing.assert_array_equal(result.to_numpy(), expected)
    np.testing.assert_array_equal(lazy_result.to_numpy(), expected)
    assert result.to_numpy().dtype == expected.dtype
    assert lazy_result.to_numpy().dtype == expected.dtype


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_trace(offset):
    arr = np.arange(12, dtype=np.int32).reshape((3, 4))

    assert finchlite.trace(arr, offset=offset).item() == np.trace(arr, offset=offset)


def test_lazy_array_api_matrix_functions():
    arr = np.arange(12, dtype=np.int32).reshape((3, 4))
    x = finchlite.lazy(arr)

    np.testing.assert_array_equal(
        finchlite.compute(finchlite.triu(x, k=1)).to_numpy(), np.triu(arr, k=1)
    )
    np.testing.assert_array_equal(
        finchlite.compute(finchlite.tril(x, k=-1)).to_numpy(), np.tril(arr, k=-1)
    )
    np.testing.assert_array_equal(
        finchlite.compute(finchlite.diag(x, k=1)).to_numpy(), np.diag(arr, k=1)
    )
    assert finchlite.compute(finchlite.trace(x, offset=1)).item() == np.trace(
        arr, offset=1
    )
