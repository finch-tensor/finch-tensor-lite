import numpy as np

import finch
import finch.finch_assembly as asm
import finch.finch_logic as log
import finch.finch_notation as ntn
from finch import ffuncs
from finch.codegen.buffers import NumpyBuffer


def create_ntn_simple_node():
    i = ntn.Variable("i", finch.int64)
    j = ntn.Variable("j", finch.int64)
    k = ntn.Variable("k", finch.int64)

    T = finch.ftype(finch.asarray(np.zeros((1, 1))))

    A = ntn.Variable("A", T)
    B = ntn.Variable("B", T)
    C = ntn.Variable("C", T)
    A_ = ntn.Slot("A_", T)
    B_ = ntn.Slot("B_", T)
    C_ = ntn.Slot("C_", T)

    a_ik = ntn.Variable("a_ik", finch.float64)
    b_kj = ntn.Variable("b_kj", finch.float64)
    c_ij = ntn.Variable("c_ij", finch.float64)

    m = ntn.Variable("m", finch.int64)
    n = ntn.Variable("n", finch.int64)
    p = ntn.Variable("p", finch.int64)

    return ntn.Module(
        (
            ntn.Function(
                ntn.Variable("matmul", T),
                (C, A, B),
                ntn.Block(
                    (
                        ntn.Unpack(A_, A),
                        ntn.Unpack(B_, B),
                        ntn.Unpack(C_, C),
                        ntn.Assign(m, ntn.Dimension(A_, ntn.Literal(0))),
                        ntn.Assign(n, ntn.Dimension(B_, ntn.Literal(1))),
                        ntn.Assign(p, ntn.Dimension(A_, ntn.Literal(1))),
                        ntn.Declare(
                            C_, ntn.Literal(0.0), ntn.Literal(ffuncs.add), (m, n)
                        ),
                        ntn.Loop(
                            i,
                            ntn.Call(
                                ntn.Literal(finch.compile.make_extent),
                                (ntn.Literal(finch.int64(0)), m),
                            ),
                            ntn.Loop(
                                k,
                                ntn.Call(
                                    ntn.Literal(finch.compile.make_extent),
                                    (ntn.Literal(finch.int64(0)), p),
                                ),
                                ntn.Loop(
                                    j,
                                    ntn.Call(
                                        ntn.Literal(finch.compile.make_extent),
                                        (ntn.Literal(finch.int64(0)), n),
                                    ),
                                    ntn.Block(
                                        (
                                            ntn.Assign(
                                                a_ik,
                                                ntn.Unwrap(
                                                    ntn.Access(A_, ntn.Read(), (i, k))
                                                ),
                                            ),
                                            ntn.Assign(
                                                b_kj,
                                                ntn.Unwrap(
                                                    ntn.Access(B_, ntn.Read(), (k, j))
                                                ),
                                            ),
                                            ntn.Assign(
                                                c_ij,
                                                ntn.Call(
                                                    ntn.Literal(ffuncs.mul),
                                                    (a_ik, b_kj),
                                                ),
                                            ),
                                            ntn.Increment(
                                                ntn.Access(
                                                    C_,
                                                    ntn.Update(ntn.Literal(ffuncs.add)),
                                                    (i, j),
                                                ),
                                                c_ij,
                                            ),
                                        )
                                    ),
                                ),
                            ),
                        ),
                        ntn.Freeze(C_, ntn.Literal(ffuncs.add)),
                        ntn.Repack(C_, C),
                        ntn.Return(C),
                    )
                ),
            ),
        )
    )


def create_log_simple_node():
    s = np.array([[2, 4], [6, 0]])
    a = np.array([[1, 2], [3, 2]])
    b = np.array([[9, 8], [6, 5]])
    i, j, k = log.Field("i"), log.Field("j"), log.Field("k")

    return log.Plan(
        (
            log.Query(log.Alias("S"), log.Table(log.Literal(s), (i, j))),
            log.Query(log.Alias("A"), log.Table(log.Literal(a), (i, k))),
            log.Query(log.Alias("B"), log.Table(log.Literal(b), (k, j))),
            log.Query(
                log.Alias("AB"),
                log.MapJoin(log.Literal(ffuncs.mul), (log.Alias("A"), log.Alias("B"))),
            ),
            # matmul
            log.Query(
                log.Alias("C"),
                log.Aggregate(
                    log.Literal(ffuncs.add), log.Literal(0), log.Alias("AB"), (k,)
                ),
            ),
            # elemwise
            log.Query(
                log.Alias("RES"),
                log.MapJoin(log.Literal(ffuncs.mul), (log.Alias("C"), log.Alias("S"))),
            ),
            log.Produces((log.Alias("RES"),)),
        )
    )


def create_asm_if_node():
    var = asm.Variable("a", finch.int64)
    return asm.Module(
        (
            asm.Function(
                asm.Variable("if_else", finch.int64),
                (),
                asm.Block(
                    (
                        asm.Assign(var, asm.Literal(np.int64(5))),
                        asm.If(
                            asm.Call(
                                asm.Literal(ffuncs.eq),
                                (var, asm.Literal(np.int64(5))),
                            ),
                            asm.Block(
                                (
                                    asm.Assign(
                                        var,
                                        asm.Call(
                                            asm.Literal(ffuncs.add),
                                            (var, asm.Literal(np.int64(10))),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        asm.IfElse(
                            asm.Call(
                                asm.Literal(ffuncs.lt),
                                (var, asm.Literal(np.int64(15))),
                            ),
                            asm.Block(
                                (
                                    asm.Assign(
                                        var,
                                        asm.Call(
                                            asm.Literal(ffuncs.sub),
                                            (var, asm.Literal(np.int64(3))),
                                        ),
                                    ),
                                )
                            ),
                            asm.Block(
                                (
                                    asm.Assign(
                                        var,
                                        asm.Call(
                                            asm.Literal(ffuncs.mul),
                                            (var, asm.Literal(np.int64(2))),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        asm.Return(var),
                    )
                ),
            ),
        )
    )


def create_asm_dot_node():
    c = asm.Variable("c", finch.float64)
    i = asm.Variable("i", finch.int64)
    ab = NumpyBuffer(np.array([1, 2, 3], dtype=np.float64))
    bb = NumpyBuffer(np.array([4, 5, 6], dtype=np.float64))
    ab_v = asm.Variable("a", ab.ftype)
    ab_slt = asm.Slot("a_", ab.ftype)
    bb_v = asm.Variable("b", bb.ftype)
    bb_slt = asm.Slot("b_", bb.ftype)

    return asm.Module(
        (
            asm.Function(
                asm.Variable("dot_product", finch.float64),
                (
                    ab_v,
                    bb_v,
                ),
                asm.Block(
                    (
                        asm.Assign(c, asm.Literal(np.float64(0.0))),
                        asm.Unpack(ab_slt, ab_v),
                        asm.Unpack(bb_slt, bb_v),
                        asm.ForLoop(
                            i,
                            asm.Literal(np.int64(0)),
                            asm.Length(ab_slt),
                            asm.Block(
                                (
                                    asm.Assign(
                                        c,
                                        asm.Call(
                                            asm.Literal(ffuncs.add),
                                            (
                                                c,
                                                asm.Call(
                                                    asm.Literal(ffuncs.mul),
                                                    (
                                                        asm.Load(ab_slt, i),
                                                        asm.Load(bb_slt, i),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        asm.Repack(ab_slt),
                        asm.Repack(bb_slt),
                        asm.Return(c),
                    )
                ),
            ),
        )
    )


def create_asm_comprehensive_node():
    a = asm.Variable("a", finch.int64)
    b = asm.Variable("b", finch.int64)
    c = asm.Variable("c", finch.int64)
    d = asm.Variable("d", finch.int64)
    result = asm.Variable("result", finch.int64)
    i = asm.Variable("i", finch.int64)
    j = asm.Variable("j", finch.int64)
    temp = asm.Variable("temp", finch.int64)

    helper_func = asm.Function(
        asm.Variable("compute", finch.int64),
        (asm.Variable("x", finch.int64), asm.Variable("y", finch.int64)),
        asm.Block(
            (
                asm.Assign(temp, asm.Variable("x", finch.int64)),
                asm.Assign(
                    temp,
                    asm.Call(
                        asm.Literal(ffuncs.add),
                        (temp, asm.Variable("y", finch.int64)),
                    ),
                ),
                asm.Return(temp),
            )
        ),
    )

    main_func = asm.Function(
        asm.Variable("main", finch.int64),
        (),
        asm.Block(
            (
                asm.Assign(a, asm.Literal(np.int64(10))),
                asm.Assign(b, a),
                asm.Assign(c, b),
                asm.Assign(d, asm.Literal(np.int64(5))),
                asm.If(
                    asm.Call(
                        asm.Literal(ffuncs.gt),
                        (c, asm.Literal(np.int64(5))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(
                                a,
                                asm.Call(
                                    asm.Literal(ffuncs.add),
                                    (a, d),
                                ),
                            ),
                            asm.Assign(b, a),
                        )
                    ),
                ),
                asm.Assign(result, asm.Literal(np.int64(0))),
                asm.ForLoop(
                    i,
                    asm.Literal(np.int64(0)),
                    asm.Literal(np.int64(5)),
                    asm.Block(
                        (
                            asm.Assign(temp, i),
                            asm.Assign(
                                result,
                                asm.Call(
                                    asm.Literal(ffuncs.add),
                                    (result, temp),
                                ),
                            ),
                            asm.If(
                                asm.Call(
                                    asm.Literal(ffuncs.eq),
                                    (
                                        asm.Call(
                                            asm.Literal(ffuncs.mod),
                                            (i, asm.Literal(np.int64(2))),
                                        ),
                                        asm.Literal(np.int64(0)),
                                    ),
                                ),
                                asm.Block(
                                    (
                                        asm.Assign(
                                            result,
                                            asm.Call(
                                                asm.Literal(ffuncs.mul),
                                                (result, asm.Literal(np.int64(2))),
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
                asm.IfElse(
                    asm.Call(
                        asm.Literal(ffuncs.gt),
                        (result, asm.Literal(np.int64(20))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, result),
                            asm.Assign(
                                result,
                                asm.Call(
                                    asm.Literal(ffuncs.add),
                                    (c, b),
                                ),
                            ),
                        )
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, b),
                            asm.Assign(
                                result,
                                asm.Call(
                                    asm.Literal(ffuncs.mul),
                                    (c, asm.Literal(np.int64(3))),
                                ),
                            ),
                        )
                    ),
                ),
                asm.ForLoop(
                    i,
                    asm.Literal(np.int64(0)),
                    asm.Literal(np.int64(3)),
                    asm.Block(
                        (
                            asm.Assign(a, i),
                            asm.ForLoop(
                                j,
                                asm.Literal(np.int64(0)),
                                asm.Literal(np.int64(2)),
                                asm.Block(
                                    (
                                        asm.Assign(b, j),
                                        asm.Assign(
                                            result,
                                            asm.Call(
                                                asm.Literal(ffuncs.add),
                                                (
                                                    result,
                                                    asm.Call(
                                                        asm.Literal(ffuncs.add),
                                                        (a, b),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
                asm.Assign(d, result),
                asm.IfElse(
                    asm.Call(
                        asm.Literal(ffuncs.lt),
                        (d, asm.Literal(np.int64(100))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, d),
                            asm.IfElse(
                                asm.Call(
                                    asm.Literal(ffuncs.gt),
                                    (c, asm.Literal(np.int64(50))),
                                ),
                                asm.Block(
                                    (
                                        asm.Assign(
                                            result,
                                            asm.Call(
                                                asm.Literal(ffuncs.mul),
                                                (c, asm.Literal(np.int64(2))),
                                            ),
                                        ),
                                    )
                                ),
                                asm.Block((asm.Assign(result, c),)),
                            ),
                        )
                    ),
                    asm.Block((asm.Assign(result, d),)),
                ),
                asm.Assign(a, result),
                asm.Assign(b, a),
                asm.Return(b),
            )
        ),
    )

    return asm.Module((helper_func, main_func))
