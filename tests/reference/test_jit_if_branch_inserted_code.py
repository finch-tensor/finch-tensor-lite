def opt_fn(A, B, use_matmul):
    if use_matmul:
        A, B = lazy((A, B))
        result = matmul(A, B)
        result, = compute((result,))
    else:
        A, B = lazy((A, B))
        result = add(A, B)
        result, = compute((result,))
    result, = lazy((result,))
    result, = compute((result,))
    return result
