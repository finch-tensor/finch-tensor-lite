"""
Example used here :
    Row 0: [0, 3, 0, 0]
    Row 1: [0, 0, 0, 0]
    Row 2: [5, 0, 2, 0]
    Row 3: [0, 0, 0, 7]

What we know:
    n_i = 4, n_j = 4
    nnz(i)  = 3  (rows 0, 2, 3)
    nnz(ij) = 4  (total)
    s_int   = 8  (bytes per int64/intp)
    s_val   = 8  (bytes per float64)

Worked out :
    dense(dense(element()))= n_i * n_j * s_val                                    = 128
    dense(sparse(element()))= (n_i+1)*s_int + nnz*s_int + nnz*s_val               = 104
    sparse(dense(element()))= 2*s_int + nnz_i*s_int + nnz_i*n_j*s_val             = 136
    sparse(sparse(element())= 2*s_int + nnz_i*s_int + (nnz_i+1)*s_int + nnz*s_int + nnz*s_val = 136
"""

import numpy as np
import pytest
from collections import OrderedDict
import finchlite as fl
from finchlite.autoschedule.tensor_stats import DCStatsFactory
from finchlite.autoschedule.galley.logical_optimizer import insert_statistics
from finchlite.finch_logic import Field, Table, Literal
from finchlite.tensor.level.dense_level import DenseLevelFType
from finchlite.tensor.level.element_level import ElementLevelFType
from finchlite.tensor.level.sparse_list_level import SparseListLevelFType


@pytest.fixture
def matrix_4x4():
    return np.array([
        [0, 3, 0, 0],
        [0, 0, 0, 0],
        [5, 0, 2, 0],
        [0, 0, 0, 7],
    ], dtype=np.float64)


@pytest.fixture
def fields_2d():
    return (Field("i"), Field("j"))


@pytest.fixture
def dc_stats(matrix_4x4, fields_2d):
    factory = DCStatsFactory()
    node = Table(Literal(fl.asarray(matrix_4x4)), fields_2d)
    stats = insert_statistics(
        stats_factory=factory,
        node=node,
        bindings=OrderedDict(),
        replace=False,
        cache={},
    )
    return stats, factory


@pytest.fixture
def elem_ftype():
    return ElementLevelFType(fill_value=np.float64(0.0))


def test_dense_dense_element(dc_stats, fields_2d, elem_ftype):
    stats, factory = dc_stats
    ftype = DenseLevelFType(_lvl_t=DenseLevelFType(_lvl_t=elem_ftype))
    cost = ftype.size_level(fields_2d, stats, factory, 1, 0)
    assert cost == 4 * 4 * 8


def test_dense_sparse_element(dc_stats, fields_2d, elem_ftype):
    stats, factory = dc_stats
    ftype = DenseLevelFType(_lvl_t=SparseListLevelFType(_lvl_t=elem_ftype))
    cost = ftype.size_level(fields_2d, stats, factory, 1, 0)
    assert cost == (4 + 1) * 8 + 4 * 8 + 4 * 8  


def test_sparse_dense_element(dc_stats, fields_2d, elem_ftype):
    stats, factory = dc_stats
    ftype = SparseListLevelFType(_lvl_t=DenseLevelFType(_lvl_t=elem_ftype))
    cost = ftype.size_level(fields_2d, stats, factory, 1, 0)
    assert cost == 2 * 8 + 3 * 8 + 3 * 4 * 8  

def test_sparse_sparse_element(dc_stats, fields_2d, elem_ftype):
    stats, factory = dc_stats
    ftype = SparseListLevelFType(_lvl_t=SparseListLevelFType(_lvl_t=elem_ftype))
    cost = ftype.size_level(fields_2d, stats, factory, 1, 0)
    assert cost == 2 * 8 + 3 * 8 + (3 + 1) * 8 + 4 * 8 + 4 * 8  

