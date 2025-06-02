import operator

import numpy as np
from numpy.testing import assert_equal

import finch
import finch.codegen.c as c
import finch.finch_assembly as asm
from finch.codegen import CCompiler


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
    b = finch.NumpyBuffer(a)
    f = finch.codegen.c.load_shared_lib(c_code).concat_buffer_with_self
    k = finch.codegen.c.CKernel(f, type(None), [finch.NumpyBuffer])
    k(b)
    result = b.arr
    expected = np.array([1, 2, 3, 2, 3, 4], dtype=np.float64)
    assert_equal(result, expected)


def test_codegen():
    a = asm.Variable("a", finch.NumpyBufferFormat(np.float64))
    i = asm.Variable("i", int)
    l = asm.Variable("l", int)
    prgm = asm.Module((
        asm.Function(
            asm.Variable("test_function", int),
            (a,),
            asm.Block(
                (
                    asm.Assign(l, asm.Length(a)),
                    asm.Resize(
                        a,
                        asm.Call(
                            asm.Immediate(operator.mul),
                            (asm.Length(a), asm.Immediate(2)),
                        ),
                    ),
                    asm.ForLoop(
                        i,
                        asm.Immediate(0),
                        l,
                        asm.Store(
                            a,
                            asm.Call(asm.Immediate(operator.add), (i, l)),
                            asm.Call(
                                asm.Immediate(operator.add),
                                (asm.Load(a, i), asm.Immediate(1)),
                            ),
                        ),
                    ),
                    asm.Return(asm.Immediate(0)),
                )
            ),
        ),
    ))
    ctx = CCompiler()
    mod = ctx(prgm)
    f = mod.test_function

    a = np.array([1, 2, 3], dtype=np.float64)
    b = finch.NumpyBuffer(a)
    f(b)
    result = b.arr
    expected = np.array([1, 2, 3, 2, 3, 4], dtype=np.float64)
    assert_equal(result, expected)




print(test_codegen())
