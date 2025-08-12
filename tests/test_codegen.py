import operator
from collections import namedtuple

import pytest

import numpy as np
from numpy.testing import assert_equal

import finch
import finch.finch_assembly as asm
from finch import ftype
from finch.codegen import (
    CCompiler,
    CGenerator,
    NumbaCompiler,
    NumbaGenerator,
    NumpyBuffer,
    NumpyBufferFType,
)


def test_add_function():
    c_code = """
    #include <stdio.h>

    int add(int a, int b) {
        return a + b;
    }
    """
    f = finch.codegen.c.load_shared_lib(c_code).add
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
    f = finch.codegen.c.load_shared_lib(c_code).concat_buffer_with_self
    k = finch.codegen.c.CKernel(f, type(None), [NumpyBufferFType(np.float64)])
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
