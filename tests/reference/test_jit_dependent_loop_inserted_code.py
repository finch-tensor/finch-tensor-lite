def opt_fn(A, n):
    A, n = maybelazy((A, n))
    B = A
    A, B = compute((A, B))
    for _i in range(getattr(sum(B), 'item')()):
        A, B = maybelazy((A, B))
        B = add(B, A)
        A, B = compute((A, B))
    B, = lazy((B,))
    B, = compute((B,))
    return B
