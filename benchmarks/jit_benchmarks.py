import finchlite as fl
from finchlite import matmul
import numpy as np
import time



fl.set_default_scheduler(ctx=fl.interface.INTERPRET_NOTATION_GALLEY)

def f1(A, B, C, D, E):
    return matmul(A, matmul(B, matmul(C, matmul(D, E))))

@fl.jit
def f1_jit(A, B, C, D, E):
    return matmul(A, matmul(B, matmul(C, matmul(D, E))))

def mat(n, m):
    return fl.asarray(np.random.randint(0, 10, (n, m)))

def mat_chain(dims: list[int]):
    return [mat(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

if __name__ == "__main__":
    dims = [1, 20, 30, 40, 50, 60]
    A, B, C, D, E = mat_chain(dims)

    print("Running f1...")
    start_time = time.time()
    result = f1(A, B, C, D, E)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")

    print("Running f1_jit...")
    start_time = time.time()
    jit_result = f1_jit(A, B, C, D, E)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    np.testing.assert_allclose(result.to_numpy(), jit_result.to_numpy())

