import numpy as np

import finch
import finch.buffer


def test_add_function():
    c_code = """
    #include <stdio.h>

    int add(int a, int b) {
        return a + b;
    }
    """
    f = finch.codegen.c.get_c_function("add", c_code)
    result = f(3, 4)
    assert result == 7, f"Expected 7, got {result}"


def test_buffer_function():

    c_code = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <stdint.h>
    #include <string.h>

    typedef struct CNumpyBuffer {
        void* data;
        size_t length;
        void* arr;
        void* (*resize)(void**, size_t);
    } CNumpyBuffer;

    void concat_buffer_with_self(struct CNumpyBuffer* buffer) {
        // Get the original data pointer and length
        double* data = (double*)(buffer->data);
        size_t length = buffer->length;

        // Resize the buffer to double its length
        buffer->data = buffer->resize(&(buffer->arr), length * 2);
        buffer->length *= 2;

        // Copy the original data to the second half of the new buffer
        for (size_t i = 0; i < length; ++i) {
            data[length + i] = data[i];
        }
    }
    """
    a = np.array([1, 2, 3], dtype=np.float64)
    b = finch.buffer.NumpyBuffer(a)
    f = finch.codegen.c.CKernel(
        "concat_buffer_with_self", c_code, [finch.buffer.NumpyBuffer]
    )
    result = f(b)
    np.assert_equal(result, np.array([1, 2, 3, 1, 2, 3], dtype=np.float64))
