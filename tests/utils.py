import pytest

try:
    import mlir  # noqa: F401
except ImportError:
    mlir = None

skip_mlir = pytest.mark.skipif(
    mlir is None, reason="MLIR Python bindings not installed."
)


def mlir_backend(func):
    return pytest.mark.mlir_backend(skip_mlir(func))
