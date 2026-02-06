import operator

import numpy as np

import finchlite.finch_assembly as asm
import finchlite.finch_logic as log
import finchlite.finch_notation as ntn
from finchlite.codegen.numpy_buffer import NumpyBuffer
from finchlite.compile import dimension


def create_ntn_simple_node() -> ntn.NotationNode:
    i = ntn.Variable("i", np.int64)
    j = ntn.Variable("j", np.int64)
    k = ntn.Variable("k", np.int64)

    A = ntn.Variable("A", np.ndarray)
    B = ntn.Variable("B", np.ndarray)
    C = ntn.Variable("C", np.ndarray)
    A_ = ntn.Slot("A_", np.ndarray)
    B_ = ntn.Slot("B_", np.ndarray)
    C_ = ntn.Slot("C_", np.ndarray)

    a_ik = ntn.Variable("a_ik", np.float64)
    b_kj = ntn.Variable("b_kj", np.float64)
    c_ij = ntn.Variable("c_ij", np.float64)

    m = ntn.Variable("m", np.int64)
    n = ntn.Variable("n", np.int64)
    p = ntn.Variable("p", np.int64)

    return ntn.Module(
        (
            ntn.Function(
                ntn.Variable("matmul", np.ndarray),
                (C, A, B),
                ntn.Block(
                    (
                        ntn.Assign(
                            m, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(0)))
                        ),
                        ntn.Assign(
                            n, ntn.Call(ntn.Literal(dimension), (B, ntn.Literal(1)))
                        ),
                        ntn.Assign(
                            p, ntn.Call(ntn.Literal(dimension), (A, ntn.Literal(1)))
                        ),
                        ntn.Unpack(A_, A),
                        ntn.Unpack(B_, B),
                        ntn.Unpack(C_, C),
                        ntn.Declare(
                            C_, ntn.Literal(0.0), ntn.Literal(operator.add), (m, n)
                        ),
                        ntn.Loop(
                            i,
                            m,
                            ntn.Loop(
                                j,
                                n,
                                ntn.Loop(
                                    k,
                                    p,
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
                                                    ntn.Literal(operator.mul),
                                                    (a_ik, b_kj),
                                                ),
                                            ),
                                            ntn.Increment(
                                                ntn.Access(
                                                    C_,
                                                    ntn.Update(
                                                        ntn.Literal(operator.add)
                                                    ),
                                                    (i, j),
                                                ),
                                                c_ij,
                                            ),
                                        )
                                    ),
                                ),
                            ),
                        ),
                        ntn.Freeze(C_, ntn.Literal(operator.add)),
                        ntn.Repack(
                            val=C_,
                            obj=C,
                        ),
                        ntn.Return(C),
                    )
                ),
            ),
        )
    )


def create_log_simple_node() -> log.LogicNode:
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
                log.MapJoin(
                    log.Literal(operator.mul), (log.Alias("A"), log.Alias("B"))
                ),
            ),
            # matmul
            log.Query(
                log.Alias("C"),
                log.Aggregate(
                    log.Literal(operator.add), log.Literal(0), log.Alias("AB"), (k,)
                ),
            ),
            # elemwise
            log.Query(
                log.Alias("RES"),
                log.MapJoin(
                    log.Literal(operator.mul), (log.Alias("C"), log.Alias("S"))
                ),
            ),
            log.Produces((log.Alias("RES"),)),
        )
    )


def create_asm_if_node() -> asm.AssemblyNode:
    var = asm.Variable("a", np.int64)
    return asm.Module(
        (
            asm.Function(
                asm.Variable("if_else", np.int64),
                (),
                asm.Block(
                    (
                        asm.Assign(var, asm.Literal(np.int64(5))),
                        asm.If(
                            asm.Call(
                                asm.Literal(operator.eq),
                                (var, asm.Literal(np.int64(5))),
                            ),
                            asm.Block(
                                (
                                    asm.Assign(
                                        var,
                                        asm.Call(
                                            asm.Literal(operator.add),
                                            (var, asm.Literal(np.int64(10))),
                                        ),
                                    ),
                                )
                            ),
                        ),
                        asm.IfElse(
                            asm.Call(
                                asm.Literal(operator.lt),
                                (var, asm.Literal(np.int64(15))),
                            ),
                            asm.Block(
                                (
                                    asm.Assign(
                                        var,
                                        asm.Call(
                                            asm.Literal(operator.sub),
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
                                            asm.Literal(operator.mul),
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


def create_asm_dot_node() -> asm.AssemblyNode:
    c = asm.Variable("c", np.float64)
    i = asm.Variable("i", np.int64)
    ab = NumpyBuffer(np.array([1, 2, 3], dtype=np.float64))
    bb = NumpyBuffer(np.array([4, 5, 6], dtype=np.float64))
    ab_v = asm.Variable("a", ab.ftype)
    ab_slt = asm.Slot("a_", ab.ftype)
    bb_v = asm.Variable("b", bb.ftype)
    bb_slt = asm.Slot("b_", bb.ftype)

    return asm.Module(
        (
            asm.Function(
                asm.Variable("dot_product", np.float64),
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
                                            asm.Literal(operator.add),
                                            (
                                                c,
                                                asm.Call(
                                                    asm.Literal(operator.mul),
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


def create_asm_comprehensive_node() -> asm.AssemblyNode:
    a = asm.Variable("a", np.int64)
    b = asm.Variable("b", np.int64)
    c = asm.Variable("c", np.int64)
    d = asm.Variable("d", np.int64)
    result = asm.Variable("result", np.int64)
    i = asm.Variable("i", np.int64)
    j = asm.Variable("j", np.int64)
    temp = asm.Variable("temp", np.int64)

    helper_func = asm.Function(
        asm.Variable("compute", np.int64),
        (asm.Variable("x", np.int64), asm.Variable("y", np.int64)),
        asm.Block(
            (
                asm.Assign(temp, asm.Variable("x", np.int64)),
                asm.Assign(
                    temp,
                    asm.Call(
                        asm.Literal(operator.add),
                        (temp, asm.Variable("y", np.int64)),
                    ),
                ),
                asm.Return(temp),
            )
        ),
    )

    main_func = asm.Function(
        asm.Variable("main", np.int64),
        (),
        asm.Block(
            (
                asm.Assign(a, asm.Literal(np.int64(10))),
                asm.Assign(b, a),
                asm.Assign(c, b),
                asm.Assign(d, asm.Literal(np.int64(5))),
                asm.If(
                    asm.Call(
                        asm.Literal(operator.gt),
                        (c, asm.Literal(np.int64(5))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(
                                a,
                                asm.Call(
                                    asm.Literal(operator.add),
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
                                    asm.Literal(operator.add),
                                    (result, temp),
                                ),
                            ),
                            asm.If(
                                asm.Call(
                                    asm.Literal(operator.eq),
                                    (
                                        asm.Call(
                                            asm.Literal(operator.mod),
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
                                                asm.Literal(operator.mul),
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
                        asm.Literal(operator.gt),
                        (result, asm.Literal(np.int64(20))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, result),
                            asm.Assign(
                                result,
                                asm.Call(
                                    asm.Literal(operator.add),
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
                                    asm.Literal(operator.mul),
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
                                                asm.Literal(operator.add),
                                                (
                                                    result,
                                                    asm.Call(
                                                        asm.Literal(operator.add),
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
                        asm.Literal(operator.lt),
                        (d, asm.Literal(np.int64(100))),
                    ),
                    asm.Block(
                        (
                            asm.Assign(c, d),
                            asm.IfElse(
                                asm.Call(
                                    asm.Literal(operator.gt),
                                    (c, asm.Literal(np.int64(50))),
                                ),
                                asm.Block(
                                    (
                                        asm.Assign(
                                            result,
                                            asm.Call(
                                                asm.Literal(operator.mul),
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
