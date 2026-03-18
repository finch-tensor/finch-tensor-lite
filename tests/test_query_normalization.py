import operator as op

from finchlite.autoschedule.galley.logical_optimizer.query_normalization import (
    merge_queries,
    preprocess_plan_for_galley,
)
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reorder,
    Table,
)


def _contains_reorder(expr) -> bool:
    if isinstance(expr, Reorder):
        return True
    if isinstance(expr, MapJoin):
        return any(_contains_reorder(arg) for arg in expr.args)
    if isinstance(expr, Aggregate):
        return _contains_reorder(expr.arg)
    return False


def test_merge_queries_inlines_alias_tables_and_keeps_produced_aliases():
    """
    INPUT:
      A1 = Table(A, i, j)
      A2 = Table(A1, j, i)
      return A2

    EXPECTED AFTER merge_queries:
      A2 = Reorder(Table(A, j, i), j, i)   # A1 inlined; direct ref to
                                               base tensor w/ reordered fields
      return A2
    """
    i = Field("i")
    j = Field("j")

    A_lit = Literal("A")
    q1 = Query(Alias("A1"), Table(A_lit, (i, j)))
    q2 = Query(Alias("A2"), Table(Alias("A1"), (j, i)))
    plan = Plan((q1, q2, Produces((Alias("A2"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    assert isinstance(preprocessed, Plan)
    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies

    assert isinstance(out_query, Query)
    assert out_query.lhs == Alias("A2")
    assert isinstance(out_query.rhs, Reorder)
    assert isinstance(out_query.rhs.arg, Table)
    assert out_query.rhs.arg.tns == A_lit
    assert out_query.rhs.arg.idxs == (j, i)

    assert isinstance(out_produces, Produces)
    assert out_produces.args == (Alias("A2"),)


def test_normalize_reorders_produces_single_outer_reorder():
    """
    INPUT:
      out = Reorder(Aggregate(add, 0, Reorder(Table(A, i, j), j, i), i), j)
      (Aggregate reduces i; inner Reorder swaps Table to (j,i); outer asks for (j,))

    EXPECTED AFTER normalize_reorders_in_plan:
      out = Reorder(Aggregate(add, 0, Table(A, i, j), i), j)
    """
    i = Field("i")
    j = Field("j")

    A_lit = Literal("A")
    inner = Reorder(Table(A_lit, (i, j)), (j, i))
    agg = Aggregate(Literal(op.add), Literal(0), inner, (i,))
    original_rhs = Reorder(agg, (j,))
    q = Query(Alias("out"), original_rhs)
    plan = Plan((q, Produces((Alias("out"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    norm_q, norm_produces = preprocessed.bodies
    assert isinstance(norm_q, Query)

    rhs = norm_q.rhs
    assert rhs.fields() == original_rhs.fields()
    assert isinstance(rhs, Reorder)
    assert rhs.idxs == (j,)

    assert isinstance(norm_produces, Produces)
    assert norm_produces.args == (Alias("out"),)


def test_preprocess_plan_for_galley_produces_canonical_queries():
    """
    INPUT:
      A1 = Aggregate(add, 0, MapJoin(mul, [Table(A,i,j), Table(B,j,k)]), j)
      # A@B -> (i,k)
      A2 = Reorder(Table(A1, i, k), i, k)   # Reorder
      return A2

    EXPECTED AFTER preprocess_plan_for_galley:
      Single merged query for A2. A1 inlined;
      A2 = Reorder(Aggregate(add, 0, MapJoin(mul,
                                      [Table(A,i,j), Table(B,j,k)]), j), i, k)
    """
    i = Field("i")
    j = Field("j")
    k = Field("k")

    A_lit = Literal("A")
    B_lit = Literal("B")

    mj = MapJoin(
        Literal(op.mul),
        (
            Table(A_lit, (i, j)),
            Table(B_lit, (j, k)),
        ),
    )
    agg = Aggregate(Literal(op.add), Literal(0), mj, (j,))
    q1 = Query(Alias("A1"), agg)
    q2 = Query(Alias("A2"), Reorder(Table(Alias("A1"), (i, k)), (i, k)))
    plan = Plan((q1, q2, Produces((Alias("A2"),))))

    preprocessed = preprocess_plan_for_galley(plan)

    # Expect a single query for A2 plus Produces(A2)
    assert isinstance(preprocessed, Plan)
    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies
    assert isinstance(out_query, Query)
    assert out_query.lhs == Alias("A2")

    rhs = out_query.rhs
    assert rhs.fields() == q2.rhs.fields()

    assert isinstance(out_produces, Produces)
    assert out_produces.args == (Alias("A2"),)


def test_merge_queries_chain_of_three_aliases():
    """
    INPUT:
      A1 = Table(A, i, j)
      A2 = Table(A1, j, i)
      A3 = Table(A2, i, j)
      return A3

    EXPECTED AFTER merge_queries:
      A3 = Reorder(Table(A, i, j), i, j)
      return A3
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    q1 = Query(Alias("A1"), Table(A_lit, (i, j)))
    q2 = Query(Alias("A2"), Table(Alias("A1"), (j, i)))
    q3 = Query(Alias("A3"), Table(Alias("A2"), (i, j)))
    plan = Plan((q1, q2, q3, Produces((Alias("A3"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies

    assert out_query.lhs == Alias("A3")
    assert isinstance(out_query.rhs, Reorder)
    assert isinstance(out_query.rhs.arg, Table)
    assert out_query.rhs.arg.tns == A_lit
    assert out_query.rhs.arg.idxs == (i, j)


def test_merge_queries_produces_multiple_aliases():
    """
    INPUT:
      A1 = Table(A, i, j)
      A2 = Table(A1, j, i)
      return A1, A2

    EXPECTED AFTER merge_queries:
      A1 = Reorder(Table(A, i, j), i, j)
      A2 = Reorder(Table(A1, j, i), j, i)   # A1 is a produced alias,
                                              so it is NOT inlined
      return A1, A2
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    q1 = Query(Alias("A1"), Table(A_lit, (i, j)))
    q2 = Query(Alias("A2"), Table(Alias("A1"), (j, i)))
    plan = Plan((q1, q2, Produces((Alias("A1"), Alias("A2")))))

    preprocessed = preprocess_plan_for_galley(plan)
    assert len(preprocessed.bodies) == 3
    out_q1, out_q2, out_produces = preprocessed.bodies

    assert out_q1.lhs == Alias("A1")
    assert isinstance(out_q1.rhs, Reorder)
    assert out_q1.rhs.idxs == (i, j)
    assert out_q1.rhs.arg == Table(A_lit, (i, j))

    assert out_q2.lhs == Alias("A2")
    assert isinstance(out_q2.rhs, Reorder)
    assert out_q2.rhs.idxs == (j, i)
    assert isinstance(out_q2.rhs.arg, Table)
    assert out_q2.rhs.arg.tns == Alias("A1")
    assert out_q2.rhs.arg.idxs == (j, i)


def test_normalize_reorders_strips_identity_reorder():
    """
    INPUT:
      out = Reorder(Table(A, i, j), j, i)   # swap i,j

    EXPECTED AFTER normalize_reorders_in_plan:
      out = Query with RHS having fields (j, i), no interior Reorders.
      (strip removes inner Reorder; single outer Reorder(Table(A,i,j), (j,i)) if needed)
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    original_rhs = Reorder(Table(A_lit, (i, j)), (j, i))
    q = Query(Alias("out"), original_rhs)
    plan = Plan((q, Produces((Alias("out"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    norm_q, _ = preprocessed.bodies

    assert norm_q.rhs.fields() == (j, i)
    assert norm_q.rhs.fields() == original_rhs.fields()


def test_normalize_reorders_nested_reorders_collapse():
    """
    INPUT:
      out = Reorder(Reorder(Table(A, i, j), j, i), i, j)   # swap then swap back

    EXPECTED AFTER normalize_reorders_in_plan:
      out = Reorder(Table(A, i, j), i, j)
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    inner = Reorder(Table(A_lit, (i, j)), (j, i))
    outer = Reorder(inner, (i, j))
    q = Query(Alias("out"), outer)
    plan = Plan((q, Produces((Alias("out"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    norm_q, _ = preprocessed.bodies

    assert norm_q.rhs.fields() == (i, j)
    assert isinstance(norm_q.rhs, Reorder)
    assert norm_q.rhs.idxs == (i, j)


def test_normalize_reorders_aggregate_with_reorder_needs_outer():
    """
    INPUT:
      out = Reorder(Aggregate(add, 0, Table(A, i, j), j), i)
      Aggregate reduces j, natural output (i,). Reorder asks for (i,) - same.

    EXPECTED AFTER normalize_reorders_in_plan:
      out = Reorder(Aggregate(add, 0, Table(A, i, j), j), i)
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    agg = Aggregate(Literal(op.add), Literal(0), Table(A_lit, (i, j)), (j,))
    original_rhs = Reorder(agg, (i,))
    q = Query(Alias("out"), original_rhs)
    plan = Plan((q, Produces((Alias("out"),))))

    preprocessed = preprocess_plan_for_galley(plan)
    norm_q, _ = preprocessed.bodies

    assert norm_q.rhs.fields() == (i,)
    assert isinstance(norm_q.rhs, Reorder)
    assert norm_q.rhs.idxs == (i,)


def test_preprocess_plan_chain_with_reorder_and_aggregate():
    """
    INPUT:
      A1 = Aggregate(add, 0, MapJoin(mul, [Table(A,i,j), Table(B,j,k)]), j)
      A2 = Reorder(Table(A1, i, k), k, i)   # swap output to (k, i)
      return A2

    EXPECTED AFTER preprocess_plan_for_galley:
      A2 = single merged query, no interior Reorders, output fields (k, i)
    """
    i = Field("i")
    j = Field("j")
    k = Field("k")
    A_lit = Literal("A")
    B_lit = Literal("B")

    mj = MapJoin(
        Literal(op.mul),
        (Table(A_lit, (i, j)), Table(B_lit, (j, k))),
    )
    agg = Aggregate(Literal(op.add), Literal(0), mj, (j,))
    q1 = Query(Alias("A1"), agg)
    q2 = Query(Alias("A2"), Reorder(Table(Alias("A1"), (i, k)), (k, i)))
    plan = Plan((q1, q2, Produces((Alias("A2"),))))

    preprocessed = preprocess_plan_for_galley(plan)

    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies
    assert out_query.lhs == Alias("A2")
    assert out_query.rhs.fields() == (k, i)
    # At most one outer Reorder, no interior Reorders
    if isinstance(out_query.rhs, Reorder):
        assert not _contains_reorder(out_query.rhs.arg)
    else:
        assert not _contains_reorder(out_query.rhs)


def test_preprocess_plan_single_table_no_change():
    """
    INPUT:
      out = Table(A, i, j)
      return out

    EXPECTED AFTER preprocess_plan_for_galley:
      out = Reorder(Table(A, i, j), i, j)
    """
    i = Field("i")
    j = Field("j")
    A_lit = Literal("A")

    q = Query(Alias("out"), Table(A_lit, (i, j)))
    plan = Plan((q, Produces((Alias("out"),))))

    preprocessed = preprocess_plan_for_galley(plan)

    assert len(preprocessed.bodies) == 2
    out_query, _ = preprocessed.bodies
    assert isinstance(out_query.rhs, Reorder)
    assert out_query.rhs.idxs == (i, j)
    assert out_query.rhs.arg == Table(A_lit, (i, j))


def test_preprocess_plan_A_at_B_at_C():
    """
    INPUT (A @ B @ C - chain of two matrix multiplications):
      A1 = Aggregate(add, 0, MapJoin(mul, [Table(A,i,j), Table(B,j,k)]), j)
      # A@B -> (i,k)
      A2 = Aggregate(add, 0, MapJoin(mul, [Table(A1,i,k), Table(C,k,l)]), k)
      # (A@B)@C -> (i,l)
      return A2

    EXPECTED AFTER preprocess_plan_for_galley:
      Single merged query for A2. push_aggregates_up distributes mul over add,
      yielding one Aggregate with reduction (j,k) over
      MapJoin(mul, [MapJoin(mul, A, B), C]).
      No interior Reorders.
    """
    i = Field("i")
    j = Field("j")
    k = Field("k")
    l_ = Field("l")
    A_lit = Literal("A")
    B_lit = Literal("B")
    C_lit = Literal("C")

    # A @ B: (i, j) @ (j, k) -> (i, k)
    ab = MapJoin(
        Literal(op.mul),
        (Table(A_lit, (i, j)), Table(B_lit, (j, k))),
    )
    q1 = Query(Alias("A1"), Aggregate(Literal(op.add), Literal(0), ab, (j,)))

    # (A @ B) @ C: (i, k) @ (k, l) -> (i, l)
    abc = MapJoin(
        Literal(op.mul),
        (Table(Alias("A1"), (i, k)), Table(C_lit, (k, l_))),
    )
    q2 = Query(Alias("A2"), Aggregate(Literal(op.add), Literal(0), abc, (k,)))
    plan = Plan((q1, q2, Produces((Alias("A2"),))))

    preprocessed = preprocess_plan_for_galley(plan)

    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies
    assert out_query.lhs == Alias("A2")
    assert out_query.rhs.fields() == (i, l_)

    # Single Aggregate with both reduction indices after push_aggregates_up.
    # Internal fields may get fresh names when inlining (e.g. j -> gensym), so
    # we only assert the count, not the exact names.
    assert isinstance(out_query.rhs, Reorder)
    assert isinstance(out_query.rhs.arg, Aggregate)
    assert len(out_query.rhs.arg.idxs) == 2


def test_merge_queries_inlines_mapjoin_aggregate_chain():
    """
    INPUT (matmul then sum over one axis):
      A = Table(X, i, i_2)
      A_2 = Table(Y, i_3, i_4)
      A_3 = MapJoin(mul, Table(A, i_11, i_12), Table(A_2, i_12, i_13))
      A_4 = Aggregate(add, 0, Table(A_3, i_16, i_17, i_18), i_17)
      return A_4

    EXPECTED AFTER merge_queries:
      A_4 = Aggregate(add, 0, MapJoin(mul, Table(X, i_16, i_17),
        Table(Y, i_17, i_18)), i_17)
      A_3 inlined into A_4; A and A_2 inlined to base tensors w/ alpha-renamed idxs.
    """
    i = Field("i")
    i_2 = Field("i_2")
    i_3 = Field("i_3")
    i_4 = Field("i_4")
    i_11 = Field("i_11")
    i_12 = Field("i_12")
    i_13 = Field("i_13")
    i_16 = Field("i_16")
    i_17 = Field("i_17")
    i_18 = Field("i_18")

    X_lit = Literal("X")
    Y_lit = Literal("Y")

    q1 = Query(Alias("A"), Table(X_lit, (i, i_2)))
    q2 = Query(Alias("A_2"), Table(Y_lit, (i_3, i_4)))
    a3_rhs = MapJoin(
        Literal(op.mul),
        (
            Table(Alias("A"), (i_11, i_12)),
            Table(Alias("A_2"), (i_12, i_13)),
        ),
    )
    q3 = Query(Alias("A_3"), a3_rhs)
    q4 = Query(
        Alias("A_4"),
        Aggregate(
            Literal(op.add),
            Literal(0),
            Table(Alias("A_3"), (i_16, i_17, i_18)),
            (i_17,),
        ),
    )
    plan = Plan((q1, q2, q3, q4, Produces((Alias("A_4"),))))

    preprocessed = preprocess_plan_for_galley(plan)

    assert isinstance(preprocessed, Plan)
    assert len(preprocessed.bodies) == 2
    out_query, out_produces = preprocessed.bodies

    assert isinstance(out_query, Query)
    assert out_query.lhs == Alias("A_4")
    assert isinstance(out_produces, Produces)
    assert out_produces.args == (Alias("A_4"),)

    rhs = out_query.rhs
    assert isinstance(rhs, Reorder)
    assert rhs.idxs == (i_16, i_18)
    rhs = rhs.arg
    assert isinstance(rhs, Aggregate)
    assert rhs.idxs == (i_17,)
    assert isinstance(rhs.arg, MapJoin)
    assert rhs.arg.args[0].tns == X_lit
    assert rhs.arg.args[0].idxs == (i_16, i_17)
    assert rhs.arg.args[1].tns == Y_lit
    assert rhs.arg.args[1].idxs == (i_17, i_18)


def _collect_reduce_idxs(expr):
    """Recursively collect all reduction indices from Aggregate nodes."""
    result = []
    if isinstance(expr, Aggregate):
        result.extend(expr.idxs)
        result.extend(_collect_reduce_idxs(expr.arg))
    elif isinstance(expr, MapJoin):
        for arg in expr.args:
            result.extend(_collect_reduce_idxs(arg))
    elif isinstance(expr, Reorder):
        result.extend(_collect_reduce_idxs(expr.arg))
    elif isinstance(expr, Table):
        pass
    return result


def test_merge_queries_same_alias_inlined_twice_unique_internal_fields():
    """
    When the same alias is inlined multiple times (e.g. B @ B where B = A @ A),
    internal fields (reduction indices) must get fresh names at each call site
    to avoid index collisions.

    INPUT:
      A = Table(X, i, j)
      B = Aggregate(add, 0, MapJoin(mul, Table(A,i,k), Table(A,k,j)), k)   # A @ A
      C = Aggregate(add, 0, MapJoin(mul, Table(B,i,m), Table(B,m,j)), m)   # B @ B
      return C

    After merge_queries, the two inlined copies of B's RHS must use distinct
    contraction indices; otherwise the computation would be wrong.
    """
    i = Field("i")
    j = Field("j")
    k = Field("k")
    m = Field("m")
    X_lit = Literal("X")

    # B = A @ A: (i,k) @ (k,j) -> (i,j), contracts over k
    a_rhs = Table(X_lit, (i, j))
    q_a = Query(Alias("A"), a_rhs)
    b_rhs = Aggregate(
        Literal(op.add),
        Literal(0),
        MapJoin(
            Literal(op.mul),
            (Table(Alias("A"), (i, k)), Table(Alias("A"), (k, j))),
        ),
        (k,),
    )
    q_b = Query(Alias("B"), b_rhs)
    # C = B @ B: (i,m) @ (m,j) -> (i,j), contracts over m
    c_rhs = Aggregate(
        Literal(op.add),
        Literal(0),
        MapJoin(
            Literal(op.mul),
            (Table(Alias("B"), (i, m)), Table(Alias("B"), (m, j))),
        ),
        (m,),
    )
    q_c = Query(Alias("C"), c_rhs)
    plan = Plan((q_a, q_b, q_c, Produces((Alias("C"),))))

    merged = merge_queries(plan)
    assert len(merged.bodies) == 2
    out_query = merged.bodies[0]
    assert isinstance(out_query, Query)
    assert out_query.lhs == Alias("C")

    reduce_idxs = _collect_reduce_idxs(out_query.rhs)
    names = [f.name for f in reduce_idxs]
    assert len(names) == len(set(names)), (
        f"Duplicate reduction index names after inlining same alias twice: {names}"
    )
