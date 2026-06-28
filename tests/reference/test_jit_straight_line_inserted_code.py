def opt_fn(A, B):
    A, B = maybelazy((A, B))
    C = matmul(A, B)
    C, = lazy((C,))
    C, = compute((C,))
    return C
