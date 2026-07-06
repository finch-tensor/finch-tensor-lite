import importlib.util
import subprocess
import sys

import pytest

import numpy as np

import finchlite as fl
from finchlite import (
    DenseLevel,
    ElementLevel,
    FiberTensor,
    NumpyBuffer,
    NumpyBufferFType,
    SparseListLevel,
    element,
    ftype,
)
from finchlite.autoschedule import with_default_scheduler
from finchlite.tensor import BufferizedNDArray


def _requires_julia_backend():
    if importlib.util.find_spec("juliapkg") is None:
        pytest.skip("juliapkg is not installed")
    if importlib.util.find_spec("juliacall") is None:
        pytest.skip("juliacall is not installed")


def test_compile_jl_import_is_lazy_and_backend_only():
    code = """
import sys
from finchlite import get_default_scheduler

before = get_default_scheduler()
import finchlite.compile_jl as compile_jl

print("juliacall" in sys.modules)
print(hasattr(compile_jl, "FinchJLTensor"))
print(compile_jl.__all__)
print(get_default_scheduler() is before)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout.splitlines() == [
        "False",
        "False",
        "['COMPILE_JULIA']",
        "True",
    ]


def test_julia_dtype_helpers_normalize_python_dtypes_without_julia():
    code = """
import sys

import numpy as np

import finchlite as fl
import finchlite.compile_jl.dtypes as dtypes

print("juliacall" in sys.modules)
print(dtypes.to_fl_dtype(np.dtype("int32")) == fl.int32)
print(dtypes.to_fl_dtype(np.int32) == fl.int32)
print("juliacall" in sys.modules)
print(hasattr(dtypes, "fl_dtype_to_jl"))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert proc.stdout.splitlines() == [
        "False",
        "True",
        "True",
        "False",
        "False",
    ]


def test_julia_kernel_converts_native_tensors(monkeypatch):
    from finchlite.compile_jl import compiler

    converted = []
    returned = []

    class FakeJL:
        class Finch:
            Tensor = "Tensor"

        def __getattr__(self, name):
            if name != "main":
                raise AttributeError(name)
            return lambda *args: "jl-result"

        def isa(self, obj, typ):
            return obj == "jl-result" and typ == self.Finch.Tensor

    def fake_tensor_to_jl(obj):
        converted.append(obj)
        return f"jl-arg-{len(converted)}"

    def fake_jl_tensor_to_python(obj):
        returned.append(obj)
        return "python-result"

    monkeypatch.setattr(compiler, "jl", FakeJL())
    monkeypatch.setattr(compiler, "tensor_to_jl", fake_tensor_to_jl)
    monkeypatch.setattr(compiler, "jl_tensor_to_python", fake_jl_tensor_to_python)

    kernel = object.__new__(compiler.FinchJLKernel)
    kernel.func_name = "main"
    kernel.jl_code = ""

    arg = BufferizedNDArray.from_numpy(np.array([1, 2], dtype=np.int64))

    assert kernel(arg) == ("python-result",)
    assert converted == [arg]
    assert returned == ["jl-result"]


def test_compile_julia_executes_native_bufferized_ndarray():
    _requires_julia_backend()
    from finchlite.compile_jl import COMPILE_JULIA

    arg = fl.asarray(np.array([1, 2, 3], dtype=np.int64))
    expr = fl.add(fl.lazy(arg), fl.lazy(arg))

    with with_default_scheduler(COMPILE_JULIA):
        result = fl.compute(expr)

    np.testing.assert_array_equal(result.to_numpy(), np.array([2, 4, 6]))


def test_compile_julia_executes_native_fiber_tensor():
    _requires_julia_backend()
    from finchlite.compile_jl import COMPILE_JULIA

    dtype = np.int64
    ptr = NumpyBuffer(np.array([0, 1, 3, 4], dtype=np.intp))
    idx = NumpyBuffer(np.array([0, 0, 1, 2], dtype=np.intp))
    data = NumpyBuffer(np.array([1, 1, 2, 1], dtype=dtype))
    elem_ftype = element(dtype(0), ftype(dtype), ftype(np.intp), NumpyBufferFType)
    arg = FiberTensor(
        DenseLevel(
            SparseListLevel(
                ElementLevel(elem_ftype, data),
                np.intp(3),
                ptr,
                idx,
            ),
            np.intp(3),
        )
    )
    expr = fl.sum(fl.lazy(arg), axis=1)

    with with_default_scheduler(COMPILE_JULIA):
        result = fl.compute(expr)

    np.testing.assert_array_equal(result.to_numpy(), np.array([1, 3, 1]))
