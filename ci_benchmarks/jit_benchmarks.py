import numpy as np

import finchlite as fl
from finchlite import matmul


@fl.jit
def _f1_jit(A, B, C, D, E):
    return matmul(A, matmul(B, matmul(C, matmul(D, E))))


class JITCompare:
    timeout = 120

    def mat(self, n, m):
        rng = np.random.default_rng()
        return fl.asarray(rng.integers(0, 10, (n, m)))

    def mat_chain(self, dims: list[int]):
        return [self.mat(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

    def setup(self):
        fl.set_default_scheduler(ctx=fl.interface.INTERPRET_NOTATION_GALLEY)
        dims = [1, 20, 30, 40, 50, 60]
        self.A, self.B, self.C, self.D, self.E = self.mat_chain(dims)

    def time_f1(self):
        return matmul(self.A, matmul(self.B, matmul(self.C, matmul(self.D, self.E))))

    def time_f1_jit(self):
        return _f1_jit(self.A, self.B, self.C, self.D, self.E)
