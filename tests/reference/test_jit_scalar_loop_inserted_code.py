def opt_fn(A, n):
    A, n = maybelazy((A, n))
    B = A
    A, B, n = compute((A, B, n))
    for _i in range(n):
        A, B, n = maybelazy((A, B, n))
        B = add(B, A)
        A, B, n = compute((A, B, n))
    B, = lazy((B,))
    B, = compute((B,))
    return B
