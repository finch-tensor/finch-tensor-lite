def opt_fn(A, B, C):
    A, B, C = maybelazy((A, B, C))
    D = matmul(A, B)
    E = add(A, C)
    F = add(D, E)
    F, = lazy((F,))
    F, = compute((F,))
    return F
