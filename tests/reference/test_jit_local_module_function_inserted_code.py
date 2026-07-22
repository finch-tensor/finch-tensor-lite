def opt_fn(A, B, n):
    A, B, n = maybelazy((A, B, n))
    C = A
    B, C, n = compute((B, C, n))
    while n > 0:
        B, C, n = maybelazy((B, C, n))
        C = getattr(finch.interface, 'add')(C, B)
        n = n - 1
        B, C, n = compute((B, C, n))
    C, = lazy((C,))
    C, = compute((C,))
    return C
