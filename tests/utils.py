from functools import wraps

import pytest


def mlir_backend(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        _ = pytest.importorskip("mlir", reason="MLIR Python bindings not installed.")
        return func(*args, **kwargs)

    return pytest.mark.mlir_backend(func_wrapper)
