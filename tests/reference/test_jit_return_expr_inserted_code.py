def opt_fn(A, B):
    A, B = lazy((A, B))
    ret = matmul(A, B)
    ret_2 = matmul(A, B)
    ret, ret_2 = compute((ret, ret_2))
    return (ret, ret_2)
