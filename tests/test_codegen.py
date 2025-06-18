import operator

import pytest

import numpy as np
from numpy.testing import assert_equal

import finch
import finch.finch_assembly as asm
from finch.codegen import (
    CCompiler,
    NumbaCompiler,
    NumpyBuffer,
    NumpyBufferFormat,
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
    k = finch.codegen.c.CKernel(f, type(None), [NumpyBufferFormat(np.float64)])
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

    a_var = asm.Variable("a", buf.format)
    i_var = asm.Variable("i", int)
    length_var = asm.Variable("l", int)
    prgm = asm.Module(
        (
            asm.Function(
                asm.Variable("test_function", int),
                (a_var,),
                asm.Block(
                    (
                        asm.Assign(length_var, asm.Length(a_var)),
                        asm.Resize(
                            a_var,
                            asm.Call(
                                asm.Literal(operator.mul),
                                (asm.Length(a_var), asm.Literal(2)),
                            ),
                        ),
                        asm.ForLoop(
                            i_var,
                            asm.Literal(0),
                            length_var,
                            asm.Store(
                                a_var,
                                asm.Call(
                                    asm.Literal(operator.add), (i_var, length_var)
                                ),
                                asm.Call(
                                    asm.Literal(operator.add),
                                    (asm.Load(a_var, i_var), asm.Literal(1)),
                                ),
                            ),
                        ),
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
    ab_v = asm.Variable("a", ab.format)
    bb_v = asm.Variable("b", bb.format)
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
                        asm.ForLoop(
                            i,
                            asm.Literal(np.int64(0)),
                            asm.Length(ab_v),
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
                                                        asm.Load(ab_v, i),
                                                        asm.Load(bb_v, i),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                )
                            ),
                        ),
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
