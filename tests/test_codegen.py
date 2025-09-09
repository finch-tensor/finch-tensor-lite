import ctypes
import operator
import re
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

from finchlite.codegen.c import construct_from_c, deserialize_from_c, serialize_to_c
import pytest

import numpy as np
from numpy.testing import assert_equal

import finchlite
from finchlite.codegen.numba_backend import construct_from_numba, deserialize_from_numba, serialize_to_numba
import finchlite.finch_assembly as asm
from finchlite import ftype
from finchlite.codegen import (
    CCompiler,
    CGenerator,
    NumbaCompiler,
    NumbaGenerator,
    NumpyBuffer,
    NumpyBufferFType,
    SafeBuffer,
)


def test_add_function():
    c_code = """
    #include <stdio.h>

    int add(int a, int b) {
        return a + b;
    }
    """
    f = finchlite.codegen.c.load_shared_lib(c_code).add
    result = f(3, 4)
    assert result == 7, f"Expected 7, got {result}"


def test_buffer_function():
    c_code = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <stdint.h>
    #include <string.h>

    typedef struct CNumpyBuffer {
        void* arr;
        void* data;
        size_t length;
        void* (*resize)(void**, size_t);
    } CNumpyBuffer;

    void concat_buffer_with_self(struct CNumpyBuffer* buffer) {
        // Get the original data pointer and length
        double* data = (double*)(buffer->data);
        size_t length = buffer->length;

        // Resize the buffer to double its length
        buffer->data = buffer->resize(&(buffer->arr), length * 2);
        buffer->length *= 2;

        // Update the data pointer after resizing
        data = (double*)(buffer->data);

        // Copy the original data to the second half of the new buffer
        for (size_t i = 0; i < length; ++i) {
            data[length + i] = data[i] + 1;
        }
    }
    """
    a = np.array([1, 2, 3], dtype=np.float64)
    b = NumpyBuffer(a)
    f = finchlite.codegen.c.load_shared_lib(c_code).concat_buffer_with_self
    k = finchlite.codegen.c.CKernel(f, type(None), [NumpyBufferFType(np.float64)])
    k(b)
    result = b.arr
    expected = np.array([1, 2, 3, 2, 3, 4], dtype=np.float64)
    assert_equal(result, expected)


@pytest.mark.parametrize(
    ["compiler", "buffer"],
    [
        (CCompiler(), NumpyBuffer),
        (NumbaCompiler(), NumpyBuffer),
    ],
)
def test_codegen(compiler, buffer):
    a = np.array([1, 2, 3], dtype=np.float64)
    buf = buffer(a)

    a_var = asm.Variable("a", buf.ftype)
    i_var = asm.Variable("i", np.intp)
    length_var = asm.Variable("l", np.intp)
    a_slt = asm.Slot("a_", buf.ftype)
    prgm = asm.Module(
        (
            asm.Function(
                asm.Variable("test_function", np.intp),
                (a_var,),
                asm.Block(
                    (
                        asm.Unpack(a_slt, a_var),
                        asm.Assign(length_var, asm.Length(a_slt)),
                        asm.Resize(
                            a_slt,
                            asm.Call(
                                asm.Literal(operator.mul),
                                (asm.Length(a_slt), asm.Literal(2)),
                            ),
                        ),
                        asm.ForLoop(
                            i_var,
                            asm.Literal(0),
                            length_var,
                            asm.Store(
                                a_slt,
                                asm.Call(
                                    asm.Literal(operator.add), (i_var, length_var)
                                ),
                                asm.Call(
                                    asm.Literal(operator.add),
                                    (asm.Load(a_slt, i_var), asm.Literal(1)),
                                ),
                            ),
                        ),
                        asm.Repack(a_slt),
                        asm.Return(asm.Literal(0)),
                    )
                ),
            ),
        )
    )
    mod = compiler(prgm)
    f = mod.test_function
    f(buf)
    result = buf.arr
    expected = np.array([1, 2, 3, 2, 3, 4], dtype=np.float64)
    assert_equal(result, expected)


@pytest.mark.parametrize(
    ["compiler", "buffer"],
    [
        (CCompiler(), NumpyBuffer),
        (NumbaCompiler(), NumpyBuffer),
        (asm.AssemblyInterpreter(), NumpyBuffer),
    ],
)
def test_dot_product(compiler, buffer):
    a = np.array([1, 2, 3], dtype=np.float64)
    b = np.array([4, 5, 6], dtype=np.float64)

    a_buf = buffer(a)
    b_buf = buffer(b)

    c = asm.Variable("c", np.float64)
    i = asm.Variable("i", np.int64)
    ab = buffer(a)
    bb = buffer(b)
    ab_v = asm.Variable("a", ab.ftype)
    ab_slt = asm.Slot("a_", ab.ftype)
    bb_v = asm.Variable("b", bb.ftype)
    bb_slt = asm.Slot("b_", bb.ftype)
    prgm = asm.Module(
        (
            asm.Function(
                asm.Variable("dot_product", np.float64),
                (
                    ab_v,
                    bb_v,
                ),
                asm.Block(
                    (
                        asm.Assign(c, asm.Literal(np.float64(0.0))),
                        asm.Unpack(ab_slt, ab_v),
                        asm.Unpack(bb_slt, bb_v),
                        asm.ForLoop(
                            i,
                            asm.Literal(np.int64(0)),
                            asm.Length(ab_slt),
                            asm.Block(
                                (
                                    asm.Assign(
                                        c,
                                        asm.Call(
                                            asm.Literal(operator.add),
                                            (
                                                c,
                                                asm.Call(
                                                    asm.Literal(operator.mul),
                                                    (
                                                        asm.Load(ab_slt, i),
                                                        asm.Load(bb_slt, i),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        asm.Repack(ab_slt),
                        asm.Repack(bb_slt),
                        asm.Return(c),
                    )
                ),
            ),
        )
    )

    mod = compiler(prgm)

    result = mod.dot_product(a_buf, b_buf)

    interp = asm.AssemblyInterpreter()(prgm)

    expected = interp.dot_product(a_buf, b_buf)

    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    ["compiler", "extension", "buffer"],
    [
        (CGenerator(), ".c", NumpyBuffer),
        (NumbaGenerator(), ".py", NumpyBuffer),
    ],
)
def test_dot_product_regression(compiler, extension, buffer, file_regression):
    a = np.array([1, 2, 3], dtype=np.float64)
    b = np.array([4, 5, 6], dtype=np.float64)

    c = asm.Variable("c", np.float64)
    i = asm.Variable("i", np.int64)
    ab = buffer(a)
    bb = buffer(b)
    ab_v = asm.Variable("a", ab.ftype)
    ab_slt = asm.Slot("a_", ab.ftype)
    bb_v = asm.Variable("b", bb.ftype)
    bb_slt = asm.Slot("b_", bb.ftype)
    prgm = asm.Module(
        (
            asm.Function(
                asm.Variable("dot_product", np.float64),
                (
                    ab_v,
                    bb_v,
                ),
                asm.Block(
                    (
                        asm.Assign(c, asm.Literal(np.float64(0.0))),
                        asm.Unpack(ab_slt, ab_v),
                        asm.Unpack(bb_slt, bb_v),
                        asm.ForLoop(
                            i,
                            asm.Literal(np.int64(0)),
                            asm.Length(ab_slt),
                            asm.Block(
                                (
                                    asm.Assign(
                                        c,
                                        asm.Call(
                                            asm.Literal(operator.add),
                                            (
                                                c,
                                                asm.Call(
                                                    asm.Literal(operator.mul),
                                                    (
                                                        asm.Load(ab_slt, i),
                                                        asm.Load(bb_slt, i),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        asm.Repack(ab_slt),
                        asm.Repack(bb_slt),
                        asm.Return(c),
                    )
                ),
            ),
        )
    )

    file_regression.check(compiler(prgm), extension=extension)


@pytest.mark.parametrize(
    ["compiler", "buffer"],
    [
        (CCompiler(), NumpyBuffer),
        (NumbaCompiler(), NumpyBuffer),
        (asm.AssemblyInterpreter(), NumpyBuffer),
    ],
)
def test_if_statement(compiler, buffer):
    var = asm.Variable("a", np.int64)
    prgm = asm.Module(
        (
            asm.Function(
                asm.Variable("if_else", np.int64),
                (),
                asm.Block(
                    (
                        asm.Assign(var, asm.Literal(np.int64(5))),
                        asm.If(
                            asm.Call(
                                asm.Literal(operator.eq),
                                (var, asm.Literal(np.int64(5))),
                            ),
                            asm.Block(
                                (
                                    asm.Assign(
                                        var,
                                        asm.Call(
                                            asm.Literal(operator.add),
                                            (var, asm.Literal(np.int64(10))),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        asm.IfElse(
                            asm.Call(
                                asm.Literal(operator.lt),
                                (var, asm.Literal(np.int64(15))),
                            ),
                            asm.Block(
                                (
                                    asm.Assign(
                                        var,
                                        asm.Call(
                                            asm.Literal(operator.sub),
                                            (var, asm.Literal(np.int64(3))),
                                        ),
                                    ),
                                )
                            ),
                            asm.Block(
                                (
                                    asm.Assign(
                                        var,
                                        asm.Call(
                                            asm.Literal(operator.mul),
                                            (var, asm.Literal(np.int64(2))),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        asm.Return(var),
                    )
                ),
            ),
        )
    )

    mod = compiler(prgm)

    result = mod.if_else()

    interp = asm.AssemblyInterpreter()(prgm)

    expected = interp.if_else()

    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "compiler",
    [
        CCompiler(),
        NumbaCompiler(),
    ],
)
def test_simple_struct(compiler):
    Point = namedtuple("Point", ["x", "y"])
    p = Point(np.float64(1.0), np.float64(2.0))
    x = (1, 4)

    p_var = asm.Variable("p", ftype(p))
    x_var = asm.Variable("x", ftype(x))
    res_var = asm.Variable("res", np.float64)
    mod = compiler(
        asm.Module(
            (
                asm.Function(
                    asm.Variable("simple_struct", np.float64),
                    (p_var, x_var),
                    asm.Block(
                        (
                            asm.Assign(
                                res_var,
                                asm.Call(
                                    asm.Literal(operator.mul),
                                    (
                                        asm.GetAttr(p_var, asm.Literal("x")),
                                        asm.GetAttr(x_var, asm.Literal("element_0")),
                                    ),
                                ),
                            ),
                            asm.Assign(
                                res_var,
                                asm.Call(
                                    asm.Literal(operator.add),
                                    (
                                        res_var,
                                        asm.Call(
                                            asm.Literal(operator.mul),
                                            (
                                                asm.GetAttr(p_var, asm.Literal("y")),
                                                asm.GetAttr(
                                                    x_var, asm.Literal("element_1")
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                            asm.Return(res_var),
                        )
                    ),
                ),
            ),
        )
    )

    result = mod.simple_struct(p, x)
    assert result == np.float64(9.0)


@pytest.mark.parametrize(
    "size,idx",
    [(size, idx) for size in range(1, 4) for idx in range(-1, 4)],
)
def test_c_load_safebuffer(size, idx):
    tester = (Path(__file__).parent / "scripts" / "safebufferaccess.py").absolute()
    result = subprocess.run(
        [
            sys.executable,
            str(tester),
            "-s",
            str(size),
            "load",
            str(idx),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if 0 <= idx < size:
        assert result.stdout.strip() == str(idx)
    else:
        assert "bounds" in result.stderr
        assert result.returncode == 1


@pytest.mark.parametrize(
    "size,idx, compiler",
    [
        (*params, compiler)
        for params in [
            (
                -1,
                2,
            ),
            (
                -1,
                3,
            ),
            (
                0,
                2,
            ),
            (
                1,
                2,
            ),
            (
                2,
                3,
            ),
            (
                2,
                2,
            ),
            (
                3,
                2,
            ),
        ]
        for compiler in [asm.AssemblyInterpreter(), NumbaCompiler()]
    ],
)
def test_numba_load_safebuffer(size, idx, compiler):
    a = np.array(range(size), dtype=np.int64)
    ab = NumpyBuffer(a)
    ab = SafeBuffer(ab)
    ab_v = asm.Variable("a", ftype(ab))
    ab_slt = asm.Slot("a_", ftype(ab))

    res_var = asm.Variable("val", ab.ftype.element_type)

    mod = compiler(
        asm.Module(
            (
                asm.Function(
                    asm.Variable("finch_access", ab.ftype.element_type),
                    (ab_v,),
                    asm.Block(
                        (
                            asm.Unpack(ab_slt, ab_v),
                            asm.Assign(
                                res_var,
                                asm.Load(ab_slt, asm.Literal(idx)),
                            ),
                            asm.Return(res_var),
                        )
                    ),
                ),
            )
        )
    )
    access = mod.finch_access
    # change = mod.finch_change
    if 0 <= idx < size:
        assert access(ab) == idx
    else:
        with pytest.raises(IndexError):
            access(ab)


@pytest.mark.parametrize(
    "size,idx,value,compiler",
    [
        (*params, compiler)
        for params in [
            (-1, 2, 3),
            (-1, 3, 1434),
            (0, 2, 3),
            (1, 2, 3),
            (2, 3, 3),
            (2, 2, 3),
            (3, 2, 3),
        ]
        for compiler in [NumbaCompiler(), asm.AssemblyInterpreter()]
    ],
)
def test_numba_store_safebuffer(size, idx, value, compiler):
    a = np.array(range(size), dtype=np.int64)
    ab = NumpyBuffer(a)
    ab = SafeBuffer(ab)
    ab_v = asm.Variable("a", ftype(ab))
    ab_slt = asm.Slot("a_", ftype(ab))

    mod = compiler(
        asm.Module(
            (
                asm.Function(
                    asm.Variable("finch_change", ab.ftype.element_type),
                    (ab_v,),
                    asm.Block(
                        (
                            asm.Unpack(ab_slt, ab_v),
                            asm.Store(
                                ab_slt,
                                asm.Literal(idx),
                                asm.Literal(value),
                            ),
                            asm.Return(asm.Load(ab_slt, asm.Literal(idx))),
                        )
                    ),
                ),
            )
        )
    )
    change = mod.finch_change
    if 0 <= idx < size:
        assert change(ab) == value
    else:
        with pytest.raises(IndexError):
            change(ab)


@pytest.mark.parametrize(
    "size,idx,value",
    [
        (*params, value)
        for params in [
            (
                -1,
                2,
            ),
            (
                -1,
                3,
            ),
            (
                0,
                2,
            ),
            (
                1,
                2,
            ),
            (
                2,
                3,
            ),
            (
                2,
                2,
            ),
            (
                3,
                2,
            ),
        ]
        for value in [-1, 1434]
    ],
)
def test_c_store_safebuffer(size, idx, value):
    tester = (Path(__file__).parent / "scripts" / "safebufferaccess.py").absolute()
    result = subprocess.run(
        [
            sys.executable,
            str(tester),
            "-s",
            str(size),
            "store",
            str(idx),
            str(value),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if 0 <= idx < size:
        arr = list(map(str, range(size)))
        arr[idx] = str(value)
        stdout = result.stdout.strip()
        stdout = re.sub(r"\s+", " ", stdout)
        stdout = stdout.replace("[ ", "[")
        assert stdout == f"[{' '.join(arr)}]"
    else:
        assert "bounds" in result.stderr
        assert result.returncode == 1

@pytest.mark.parametrize(
    "value,np_type,c_type",
    [
        (3, np.int64, ctypes.c_int64),
        (2, np.int32, ctypes.c_int32),
        (1, np.float32, ctypes.c_float),
        (1.0, np.float64, ctypes.c_double),
        (1.2, np.float64, ctypes.c_double),
    ]
)
def test_np_c_serialization(value, np_type, c_type):
    serialized = serialize_to_c(np_type, np_type(value))
    assert serialized.value == c_type(value).value
    assert isinstance(serialized, c_type)
    constructed = construct_from_c(np_type, serialized)
    assert constructed == np_type(value)
    assert deserialize_from_c(np_type, constructed, serialized) is None

@pytest.mark.parametrize(
    "value,c_type",
    [
        (3, ctypes.c_int64),
        (2, ctypes.c_int32),
        (1, ctypes.c_float),
        (1.0, ctypes.c_double),
        (1.2, ctypes.c_double),
    ]
)
def test_ctypes_c_serialization(value, c_type):
    cvalue = c_type(value)
    serialized = serialize_to_c(c_type, cvalue)
    assert serialized.value == c_type(value).value
    assert isinstance(serialized, c_type)
    constructed = construct_from_c(c_type, serialized)
    assert constructed.value == c_type(value).value
    assert deserialize_from_c(c_type, constructed, serialized) is None

@pytest.mark.parametrize(
    "value,np_type",
    [
        (3, np.int64),
        (2, np.int32),
        (1, np.float32),
        (1.0, np.float64),
        (1.2, np.float64),
    ]
)
def test_np_numba_serialization(value, np_type):
    cvalue = np_type(value)
    serialized = serialize_to_numba(np_type, cvalue)
    assert serialized == np_type(value)
    assert isinstance(serialized, np_type)
    constructed = construct_from_numba(np_type, serialized)
    assert constructed == np_type(value)
    assert deserialize_from_numba(np_type, constructed, serialized) is None
