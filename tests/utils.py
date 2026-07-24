import pytest

try:
    import mlir  # noqa: F401

    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False

skip_mlir = pytest.mark.skipif(
    not MLIR_AVAILABLE, reason="MLIR Python bindings not installed."
)


def mlir_backend(func):
    return pytest.mark.mlir_backend(skip_mlir(func))
