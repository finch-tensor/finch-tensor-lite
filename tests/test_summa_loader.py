import numpy as np

from finchlite.finch_einsum import (
    Alias,
    Collective,
    DistributedFinchTensor,
    SummaEinsumLoader,
    parse_einop,
)

from .conftest import finch_assert_allclose


class _AnyTensorFType:
    def __init__(self, ndim: int):
        self.ndim = ndim

    def fisinstance(self, _other):
        return True


def _matmul_bindings(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    return {
        Alias("A"): _AnyTensorFType(a.ndim),
        Alias("B"): _AnyTensorFType(b.ndim),
        Alias("C"): _AnyTensorFType(c.ndim),
    }


def test_summa_loader_matmul_overwrite():
    rng = np.random.default_rng(0)
    a = rng.random((5, 4))
    b = rng.random((4, 3))
    c = np.zeros((5, 3), dtype=np.result_type(a.dtype, b.dtype))

    prgm = parse_einop("C[i,j] = A[i,k] * B[k,j]")
    lib, _, _ = SummaEinsumLoader(block_k=2)(prgm, _matmul_bindings(a, b, c))

    (out,) = lib.main(a, b, c)

    finch_assert_allclose(out, a @ b)
    finch_assert_allclose(c, a @ b)


def test_summa_loader_matmul_add_reduction():
    rng = np.random.default_rng(1)
    a = rng.random((4, 6))
    b = rng.random((6, 2))
    c0 = rng.random((4, 2))
    c = c0.copy()

    prgm = parse_einop("C[i,j] += A[i,k] * B[k,j]")
    lib, _, _ = SummaEinsumLoader(block_k=3)(prgm, _matmul_bindings(a, b, c))

    (out,) = lib.main(a, b, c)

    finch_assert_allclose(out, c0 + (a @ b))
    finch_assert_allclose(c, c0 + (a @ b))


def test_summa_loader_distributed_tensor_inputs():
    rng = np.random.default_rng(2)
    collective = Collective(rank=0, size=1)

    a = DistributedFinchTensor(rng.random((3, 3)), collective)
    b = DistributedFinchTensor(rng.random((3, 4)), collective)
    c = DistributedFinchTensor(np.zeros((3, 4)), collective)

    prgm = parse_einop("C[i,j] = A[i,k] * B[k,j]")
    bindings = _matmul_bindings(a.local, b.local, c.local)
    lib, _, _ = SummaEinsumLoader(collective=collective, block_k=2)(prgm, bindings)

    (out,) = lib.main(a, b, c)

    assert isinstance(out, DistributedFinchTensor)
    finch_assert_allclose(out.local, a.local @ b.local)
    finch_assert_allclose(c.local, a.local @ b.local)
