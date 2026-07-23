"""
Shared functionality across TensorStats implementations.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from finch import finch_notation as ntn
from finch.algebra import ffuncs, ftype, int64
from finch.compile import make_extent
from finch.finch_logic import Field
from finch.tensor import BufferizedNDArray

_INT64_VECTOR_FTYPE = BufferizedNDArray.from_numpy(np.zeros(1, dtype=np.int64)).ftype


def _int_tuple_ftype(size: int):
    return ftype(tuple(np.int64(0) for _ in range(size)))


def degree_count_scan(
    arr: Any, fields: Iterable[Field], fill_value: Any
) -> tuple[list[np.ndarray], int]:
    """Scan ``arr`` and return per-dimension degree sequences and ``nnz``.

    Args:
        arr: The tensor to scan.
        fields: The axis names of ``arr`` (one per dimension), in order.
        fill_value: The value treated as "empty"; entries not equal to it are
            counted.

    Returns:
        A pair ``(counts, nnz)`` where ``counts[i]`` is a 1-D ``int64`` numpy
        array of length ``arr.shape[i]`` giving, for each index of dimension
        ``i``, the number of non-fill entries with that index, and ``nnz`` is
        the total number of non-fill entries.
    """
    fields = list(fields)
    ndims = len(fields)

    dim_loop_variables = [ntn.Variable(f"{fields[i]}", int64) for i in range(ndims)]
    dim_array_variables = [
        ntn.Variable(f"x_{fields[i]}", _INT64_VECTOR_FTYPE) for i in range(ndims)
    ]
    dim_size_variables = [ntn.Variable(f"n_{fields[i]}", int64) for i in range(ndims)]
    dim_array_slots = [
        ntn.Slot(f"x_{fields[i]}_", _INT64_VECTOR_FTYPE) for i in range(ndims)
    ]

    A = ntn.Variable("A", arr.ftype)
    A_ = ntn.Slot("A_", arr.ftype)
    A_access = ntn.Unwrap(ntn.Access(A_, ntn.Read(), tuple(dim_loop_variables)))
    A_nnz_variable = ntn.Variable("nnz", int64)

    dim_size_assignments = []
    dim_array_unpacks = []
    dim_array_declares = []
    dim_array_increments = []
    for i in range(ndims):
        dim_size_assignments.append(
            ntn.Assign(dim_size_variables[i], ntn.Dimension(A_, ntn.Literal(i)))
        )
        dim_array_unpacks.append(ntn.Unpack(dim_array_slots[i], dim_array_variables[i]))
        dim_array_declares.append(
            ntn.Declare(
                dim_array_slots[i],
                ntn.Literal(int64(0)),
                ntn.Literal(ffuncs.add),
                (dim_size_variables[i],),
            )
        )
        dim_array_increments.append(
            ntn.Increment(
                ntn.Access(
                    dim_array_slots[i],
                    ntn.Update(ntn.Literal(ffuncs.add)),
                    (dim_loop_variables[i],),
                ),
                ntn.Call(ntn.Literal(ffuncs.ne), (A_access, ntn.Literal(fill_value))),
            )
        )

    array_build_loop: ntn.NotationStatement = ntn.Block(
        (
            *dim_array_increments,
            ntn.Assign(
                A_nnz_variable,
                ntn.Call(
                    ntn.Literal(ffuncs.add),
                    (
                        A_nnz_variable,
                        ntn.Call(
                            ntn.Literal(ffuncs.ne),
                            (A_access, ntn.Literal(fill_value)),
                        ),
                    ),
                ),
            ),
        )
    )
    for i in range(ndims):
        array_build_loop = ntn.Loop(
            dim_loop_variables[i],
            ntn.Call(
                ntn.Literal(make_extent),
                (ntn.Literal(int64(0)), dim_size_variables[i]),
            ),
            array_build_loop,
        )

    dim_array_freezes = [
        ntn.Freeze(dim_array_slots[i], ntn.Literal(ffuncs.add)) for i in range(ndims)
    ]
    dim_array_repacks = [
        ntn.Repack(dim_array_slots[i], dim_array_variables[i]) for i in range(ndims)
    ]

    def to_tuple(*args):
        return (*args,)

    return_expr = ntn.Return(ntn.Call(ntn.Literal(to_tuple), (A_nnz_variable,)))

    prgm = ntn.Module(
        (
            ntn.Function(
                ntn.Variable("degree_count_scan", _int_tuple_ftype(1)),
                (A, *dim_array_variables),
                ntn.Block(
                    (
                        ntn.Unpack(A_, A),
                        *dim_size_assignments,
                        ntn.Assign(A_nnz_variable, ntn.Literal(int64(0))),
                        *dim_array_unpacks,
                        *dim_array_declares,
                        array_build_loop,
                        *dim_array_freezes,
                        *dim_array_repacks,
                        ntn.Repack(A_, A),
                        return_expr,
                    )
                ),
            ),
        )
    )
    mod = ntn.NotationInterpreter()(prgm)

    dim_array_instances = [
        BufferizedNDArray.from_numpy(np.zeros(arr.shape[i], dtype=np.int64))
        for i in range(ndims)
    ]
    out = mod.degree_count_scan(arr, *dim_array_instances)
    counts = [inst.to_numpy().copy() for inst in dim_array_instances]
    return counts, int(out[0])
