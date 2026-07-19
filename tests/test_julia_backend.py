import importlib.util

import pytest

import numpy as np

import finchlite as fl
from finchlite import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    NumpyBuffer,
    NumpyBufferFType,
    SparseByteMapLevel,
    SparseCOOLevel,
    SparseHashLevel,
    SparseListLevel,
    element,
    ftype,
)
from finchlite.autoschedule import (
    DefaultLogicOptimizer,
    DefaultLoopOrderer,
    FDFormatter,
    LogicCompiler,
    LogicExecutor,
    LogicNormalizer,
    with_default_scheduler,
)
from finchlite.autoschedule.tensor_stats import FDStatsFactory

DTYPE = np.int64
ROWS = np.intp(3)
COLS = np.intp(3)
ROW_PTR = NumpyBuffer(np.array([0, 2, 3, 5], dtype=np.intp))
COL_IDX = NumpyBuffer(np.array([0, 2, 1, 0, 2], dtype=np.intp))
STORED_VALUES = np.array([1, 2, 3, 4, 5], dtype=DTYPE)
EXPECTED_ROW_SUMS = np.array([3, 3, 9], dtype=DTYPE)


def _requires_julia_backend():
    if importlib.util.find_spec("juliapkg") is None:
        pytest.skip("juliapkg is not installed")
    if importlib.util.find_spec("juliacall") is None:
        pytest.skip("juliacall is not installed")


def _element_level(data) -> ElementLevel:
    elem_ftype = element(DTYPE(0), ftype(DTYPE), ftype(np.intp), NumpyBufferFType)
    return ElementLevel(elem_ftype, NumpyBuffer(np.asarray(data, dtype=DTYPE)))


def _compute_sparse_axis_sum(level):
    from finchlite.compile_jl import COMPILE_JULIA

    arg = FiberTensor(DenseLevel(level, ROWS))
    expr = fl.sum(fl.lazy(arg), axis=1)

    with with_default_scheduler(COMPILE_JULIA):
        return fl.compute(expr)


def _compile_julia_fd(formatter):
    return LogicNormalizer(
        LogicExecutor(
            DefaultLogicOptimizer(DefaultLoopOrderer(formatter)),
            stats_factory=FDStatsFactory(),
        )
    )


def test_compile_julia_sums_sparse_list_level():
    _requires_julia_backend()
    level = SparseListLevel(
        _element_level(STORED_VALUES),
        COLS,
        ROW_PTR,
        COL_IDX,
    )

    result = _compute_sparse_axis_sum(level)

    np.testing.assert_array_equal(result.to_numpy(), EXPECTED_ROW_SUMS)


def test_compile_julia_sums_sparse_coo_level():
    _requires_julia_backend()
    level = SparseCOOLevel(
        _element_level(STORED_VALUES),
        (COLS,),
        ROW_PTR,
        (COL_IDX,),
    )

    result = _compute_sparse_axis_sum(level)

    np.testing.assert_array_equal(result.to_numpy(), EXPECTED_ROW_SUMS)


def test_compile_julia_sums_sparse_bytemap_level():
    _requires_julia_backend()
    stored_positions = np.array([0, 2, 4, 6, 8], dtype=np.intp)
    table = np.zeros(9, dtype=np.bool_)
    table[stored_positions] = True
    data = np.array([1, 0, 2, 0, 3, 0, 4, 0, 5], dtype=DTYPE)
    level = SparseByteMapLevel(
        _element_level(data),
        COLS,
        ROW_PTR,
        NumpyBuffer(table),
        NumpyBuffer(stored_positions),
    )

    result = _compute_sparse_axis_sum(level)

    np.testing.assert_array_equal(result.to_numpy(), EXPECTED_ROW_SUMS)


def test_compile_julia_sums_sparse_hash_level():
    _requires_julia_backend()
    entry_dtype = np.dtype(
        [
            ("element_0", np.intp),
            ("element_1", np.intp),
            ("element_2", np.intp),
        ]
    )
    entries = np.array(
        [(0, 0, 0), (0, 2, 1), (1, 1, 2), (2, 0, 3), (2, 2, 4)],
        dtype=entry_dtype,
    )
    level = SparseHashLevel(
        _element_level(STORED_VALUES),
        COLS,
        ROW_PTR,
        NumpyBuffer(np.full(len(entries), 0x80, dtype=np.uint8)),
        NumpyBuffer(entries),
        NumpyBuffer(np.array([], dtype=np.intp)),
        NumpyBuffer(np.arange(len(entries), dtype=np.intp)),
        single_writer=False,
    )

    result = _compute_sparse_axis_sum(level)

    np.testing.assert_array_equal(result.to_numpy(), EXPECTED_ROW_SUMS)


def test_compile_julia_with_fd_formatter_uses_dense_output_levels():
    _requires_julia_backend()
    from finchlite.compile_jl.compiler import FinchJLCompiler

    class RecordingFDFormatter(FDFormatter):
        def __init__(self, loader):
            super().__init__(loader)
            self.output_ftypes = []

        def get_tensor_ftype(self, fill_value, shape_type, stats):
            tensor_ftype = super().get_tensor_ftype(fill_value, shape_type, stats)
            self.output_ftypes.append(tensor_ftype)
            return tensor_ftype

    formatter = RecordingFDFormatter(LogicCompiler(FinchJLCompiler()))
    scheduler = _compile_julia_fd(formatter)
    data = np.array([[1, 0, 2], [0, 3, 4]], dtype=DTYPE)
    arg = fl.asarray(data)
    expr = fl.lazy(arg) + fl.lazy(arg)

    with with_default_scheduler(scheduler):
        result = fl.compute(expr)

    np.testing.assert_array_equal(result.to_numpy(), data + data)
    assert formatter.output_ftypes
    output_ftype = formatter.output_ftypes[-1]
    assert isinstance(output_ftype, fl.FiberTensorFType)
    assert isinstance(output_ftype.lvl_t, fl.DenseLevelFType)
    assert isinstance(output_ftype.lvl_t.lvl_t, fl.DenseLevelFType)
    assert isinstance(output_ftype.lvl_t.lvl_t.lvl_t, fl.ElementLevelFType)
