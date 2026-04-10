from collections import OrderedDict

import pytest

import numpy as np

import finchlite as fl
from finchlite import ffuncs
from finchlite.autoschedule.galley.logical_optimizer import (
    AnnotatedQuery,
    greedy_query,
    insert_statistics,
)
from finchlite.autoschedule.tensor_stats import DenseStatsFactory
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    MapJoin,
    Query,
    Table,
)

A = fl.asarray(np.ones((2,)))
A_mat = fl.asarray(np.ones((5, 5)))
B = fl.asarray(np.ones((3,)))
C_mat = fl.asarray(np.ones((2, 4)))
D = fl.asarray(np.ones((4,)))


# ─────────────────────────────── Annotated_Query tests ─────────────────────────────
@pytest.mark.parametrize(
    "reduce_idxs,parent_idxs,expected",
    [
        # Some indices have parents
        (["i", "j", "k"], {"i": [], "j": ["i"], "k": []}, ["i", "k"]),
        # Keys missing from parent map should be treated as zero parents.
        (["i", "j", "k"], {"j": ["i"]}, ["i", "k"]),
        # All have parents
        (["a", "b"], {"a": ["b"], "b": ["a"]}, []),
        # Empty input
        ([], {}, []),
        # Order preserved among reducible indices
        (["x", "y", "z"], {"y": ["x"]}, ["x", "z"]),
    ],
)
def test_get_reducible_idxs(reduce_idxs, parent_idxs, expected):
    names = set(reduce_idxs)
    names.update(parent_idxs.keys())
    for i in parent_idxs.values():
        names.update(i)

    fields: dict[str, Field] = {x: Field(x) for x in names}
    reduce_fields: list[Field] = [fields[name] for name in reduce_idxs]
    parent_fields: OrderedDict[Field, list[Field]] = OrderedDict(
        (fields[key], [fields[p] for p in parents])
        for key, parents in parent_idxs.items()
    )

    aq = object.__new__(AnnotatedQuery)
    aq.stats_factory = object
    aq.output_name = None
    aq.reduce_idxs = reduce_fields
    aq.point_expr = None
    aq.idx_lowest_root = OrderedDict()
    aq.idx_op = OrderedDict()
    aq.idx_init = OrderedDict()
    aq.parent_idxs = parent_fields
    aq.original_idx = OrderedDict()
    aq.connected_components = []
    aq.connected_idxs = OrderedDict()
    aq.output_order = None
    aq.output_format = None
    aq.bindings = OrderedDict()

    result = [field.name for field in AnnotatedQuery.get_reducible_idxs(aq)]
    assert result == expected


@pytest.mark.parametrize(
    "reduce_idxs,parent_idxs,component,expected",
    [
        # Component {i,k} with i,k reducible; j has parent i
        (["i", "j", "k"], {"j": ["i"]}, ["i", "j", "k"], ["i", "k"]),
        # Component {i,j} - only i reducible (j has parent i)
        (["i", "j", "k"], {"j": ["i"]}, ["i", "j"], ["i"]),
        # Component {k} - k reducible, k not in component {i,j}
        (["i", "j", "k"], {"j": ["i"]}, ["k"], ["k"]),
        # Two components: {a,b} and {c}; in {a,b} only a reducible
        (["a", "b", "c"], {"b": ["a"]}, ["a", "b"], ["a"]),
        (["a", "b", "c"], {"b": ["a"]}, ["c"], ["c"]),
        # Empty component
        (["a", "b"], {"b": ["a"]}, [], []),
    ],
)
def test_get_reducible_idxs_for_component(
    reduce_idxs, parent_idxs, component, expected
):
    names = set(reduce_idxs) | set(parent_idxs.keys())
    for parents in parent_idxs.values():
        names.update(parents)

    fields: dict[str, Field] = {x: Field(x) for x in names}
    reduce_fields: list[Field] = [fields[name] for name in reduce_idxs]
    component_fields: list[Field] = [fields[name] for name in component]
    parent_fields: OrderedDict[Field, list[Field]] = OrderedDict(
        (fields[key], [fields[p] for p in parents])
        for key, parents in parent_idxs.items()
    )

    aq = object.__new__(AnnotatedQuery)
    aq.stats_factory = object
    aq.output_name = None
    aq.reduce_idxs = reduce_fields
    aq.point_expr = None
    aq.idx_lowest_root = OrderedDict()
    aq.idx_op = OrderedDict()
    aq.idx_init = OrderedDict()
    aq.parent_idxs = parent_fields
    aq.original_idx = OrderedDict()
    aq.connected_components = []
    aq.connected_idxs = OrderedDict()
    aq.output_order = None
    aq.output_format = None
    aq.bindings = OrderedDict()

    result = [
        field.name for field in aq.get_reducible_idxs_for_component(component_fields)
    ]
    assert result == expected


@pytest.mark.parametrize(
    "parent_idxs, connected_idxs, expected",
    [
        # Single component; order within component follows connected_idxs key order
        (
            {},
            {"a": ["b"], "b": ["a"]},
            [["a", "b"]],
        ),
        # Two components: {a,b} and {c}
        (
            {},
            {"a": ["b"], "b": ["a"], "c": []},
            [["a", "b"], ["c"]],
        ),
        # Parent edge is ignored for connectivity
        (
            {"b": ["a"]},
            {"a": ["b"], "b": ["a"]},
            [["a"], ["b"]],
        ),
        # Ordering across components is enforced
        (
            {"b": ["a"]},
            {"b": [], "a": []},
            [["a"], ["b"]],
        ),
        # Chain of three separate components with parents
        (
            {"b": ["a"], "c": ["b"]},
            {"c": [], "b": [], "a": []},
            [["a"], ["b"], ["c"]],
        ),
        # Single big component
        (
            {"b": ["a"], "c": ["b"]},
            {"a": ["b"], "b": ["a", "c"], "c": ["b"]},
            [["a"], ["b"], ["c"]],
        ),
    ],
)
def test_get_idx_connected_components(parent_idxs, connected_idxs, expected):
    names: set[str] = set(parent_idxs.keys()) | set(connected_idxs.keys())
    for i in parent_idxs.values():
        names.update(i)
    for j in connected_idxs.values():
        names.update(j)

    name = {x: Field(x) for x in names}

    parent_field_idxs: dict[Field, list[Field]] = {
        name[k]: [name[p] for p in v] for k, v in parent_idxs.items()
    }
    connected_field_idxs: dict[Field, list[Field]] = {
        name[k]: [name[n] for n in v] for k, v in connected_idxs.items()
    }

    components = AnnotatedQuery.get_idx_connected_components(
        parent_field_idxs, connected_field_idxs
    )
    result = [[field.name for field in comp] for comp in components]

    assert result == expected


@pytest.mark.parametrize(
    "expr,node_to_replace,new_node,nodes_to_remove,expected_names",
    [
        (
            MapJoin(
                Literal("op"),
                (
                    Table(Literal("a"), (Field("a"),)),
                    Table(Literal("b"), (Field("b"),)),
                    Table(Literal("c"), (Field("c"),)),
                ),
            ),
            Table(Literal("b"), (Field("b"),)),
            Table(Literal("a"), (Field("a"),)),
            set(),
            ["a", "a", "c"],
        ),
        (
            MapJoin(
                Literal("op"),
                (
                    Table(Literal("a"), (Field("a"),)),
                    Table(Literal("b"), (Field("b"),)),
                    Table(Literal("c"), (Field("c"),)),
                ),
            ),
            Table(Literal("b"), (Field("b"),)),
            Table(Literal("a"), (Field("a"),)),
            {Table(Literal("c"), (Field("c"),))},
            ["a", "a"],
        ),
        (
            MapJoin(
                Literal("op"),
                (
                    Table(Literal("a"), (Field("a"),)),
                    Table(Literal("b"), (Field("b"),)),
                    Table(Literal("c"), (Field("c"),)),
                ),
            ),
            Table(Literal("c"), (Field("c"),)),
            Table(Literal("a"), (Field("a"),)),
            {Table(Literal("c"), (Field("c"),))},
            ["a", "b"],
        ),
        (
            MapJoin(
                Literal("op"),
                (
                    Table(Literal("a"), (Field("a"),)),
                    Table(Literal("b"), (Field("b"),)),
                    Table(Literal("c"), (Field("c"),)),
                ),
            ),
            Table(Literal("b"), (Field("b"),)),
            Table(Literal("a"), (Field("a"),)),
            {Table(Literal("b"), (Field("b"),))},
            ["a", "c"],
        ),
        (
            MapJoin(
                Literal("op"),
                (
                    Table(Literal("a"), (Field("a"),)),
                    Table(Literal("b"), (Field("b"),)),
                    Table(Literal("c"), (Field("c"),)),
                ),
            ),
            Table(Literal("c"), (Field("c"),)),
            Table(Literal("a"), (Field("a"),)),
            set(),
            ["a", "b", "a"],
        ),
    ],
)
def test_replace_and_remove_nodes(
    expr,
    node_to_replace,
    new_node,
    nodes_to_remove,
    expected_names,
):
    out = AnnotatedQuery.replace_and_remove_nodes(
        expr=expr,
        node_to_replace=node_to_replace,
        new_node=new_node,
        nodes_to_remove=nodes_to_remove,
    )

    result = [tbl.idxs[0].name for tbl in out.args]
    assert result == expected_names


@pytest.mark.parametrize(
    "root, idx_name, expected",
    [
        # Distributive case:
        # root = MapJoin(mul, [A(i), B(j)]), reduce over j → [B]
        (
            MapJoin(
                Literal(ffuncs.mul),
                (
                    Table(Literal("A"), (Field("i"),)),
                    Table(Literal("B"), (Field("j"),)),
                ),
            ),
            "j",
            ["B"],
        ),
        # Split-push case:
        # root = MapJoin(add, [A(i), B(i), C(j)]), reduce over i → [C, A, B]
        (
            MapJoin(
                Literal(ffuncs.add),
                (
                    Table(Literal("A"), (Field("i"),)),
                    Table(Literal("B"), (Field("i"),)),
                    Table(Literal("C"), (Field("j"),)),
                ),
            ),
            "i",
            ["C", "A", "B"],
        ),
        # Leaf case:
        # root = Table(A(i)), reduce over i → [A]
        (
            Table(Literal("A"), (Field("i"),)),
            "i",
            ["A"],
        ),
        # Nested case:
        # root = MapJoin(mul, [A(i,j), B(j)]), reduce over i → [A]
        (
            MapJoin(
                Literal(ffuncs.mul),
                (
                    Table(Literal("A"), (Field("i"), Field("j"))),
                    Table(Literal("B"), (Field("j"),)),
                ),
            ),
            "i",
            ["A"],
        ),
        # Special case: max(C(i), D(j)), reduce over i → [max(C,D)]
        (
            MapJoin(
                Literal(ffuncs.max),
                (
                    Table(Literal("C"), (Field("i"),)),
                    Table(Literal("D"), (Field("j"),)),
                ),
            ),
            "i",
            [
                MapJoin(
                    Literal(ffuncs.max),
                    (
                        Table(Literal("C"), (Field("i"),)),
                        Table(Literal("D"), (Field("j"),)),
                    ),
                )
            ],
        ),
        # root = MapJoin(mul, [A(j), MapJoin(max, [B(i), C(j)])]), reduce over i
        (
            MapJoin(
                Literal(ffuncs.mul),
                (
                    Table(Literal("A"), (Field("j"),)),
                    MapJoin(
                        Literal(ffuncs.max),
                        (
                            Table(Literal("B"), (Field("i"),)),
                            Table(Literal("C"), (Field("j"),)),
                        ),
                    ),
                ),
            ),
            "i",
            [
                MapJoin(
                    Literal(ffuncs.max),
                    (
                        Table(Literal("B"), (Field("i"),)),
                        Table(Literal("C"), (Field("j"),)),
                    ),
                )
            ],
        ),
    ],
)
def test_find_lowest_roots(root, idx_name, expected):
    roots = AnnotatedQuery.find_lowest_roots(ffuncs.add, Field(idx_name), root)

    # Special-case: the max(C(i), D(j)) example – we expect the MapJoin itself.
    if expected and not isinstance(expected[0], str):
        assert roots == expected
    else:
        # All other cases:
        result: list[str] = []
        for node in roots:
            assert isinstance(node, Table)
            assert isinstance(node.tns, Literal)
            result.append(node.tns.val)

        assert result == expected


@pytest.mark.parametrize(
    ("expr", "reduce_field", "expected"),
    [
        (
            # Case 1: expr = sum_i A[i] * B[j]
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(B), (Field("j"),)),
                    ),
                ),
                (Field("i"),),
            ),
            Field("i"),
            # expected: sum_i A[i]
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(Literal(A), (Field("i"),)),),
                ),
                (Field("i"),),
            ),
        ),
        (
            # Case 2: expr = sum_i A[i] * A[i]
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(A), (Field("i"),)),
                    ),
                ),
                (Field("i"),),
            ),
            Field("i"),
            # expected: sum_i (A[i] * A[i])
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(A), (Field("i"),)),
                    ),
                ),
                (Field("i"),),
            ),
        ),
        (
            # Case 3: expr = sum_i A[i] * C[i,k] * B[j]
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(C_mat), (Field("i"), Field("k"))),
                        Table(Literal(B), (Field("j"),)),
                    ),
                ),
                (Field("i"),),
            ),
            Field("i"),
            # expected: sum_i (A[i] * C[i,k])
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(C_mat), (Field("i"), Field("k"))),
                    ),
                ),
                (Field("i"),),
            ),
        ),
        (
            # Case 4: expr = sum_i A[i] * C[i,k] * D[k] * B[j]
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(C_mat), (Field("i"), Field("k"))),
                        Table(Literal(D), (Field("k"),)),
                        Table(Literal(B), (Field("j"),)),
                    ),
                ),
                (Field("i"),),
            ),
            Field("i"),
            # expected: sum_i (A[i] * C[i,k] * D[k])
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(C_mat), (Field("i"), Field("k"))),
                        Table(Literal(D), (Field("k"),)),
                    ),
                ),
                (Field("i"),),
            ),
        ),
    ],
)
def test_get_reduce_query(expr, reduce_field, expected):
    aq = object.__new__(AnnotatedQuery)
    aq.stats_factory = DenseStatsFactory()
    aq.output_name = None
    aq.reduce_idxs = [reduce_field]
    aq.point_expr = expr
    aq.idx_lowest_root = OrderedDict({reduce_field: expr.arg})
    aq.idx_op = OrderedDict({reduce_field: ffuncs.add})
    aq.idx_init = OrderedDict({reduce_field: 0})
    aq.parent_idxs = OrderedDict()
    aq.original_idx = OrderedDict({reduce_field: reduce_field})
    aq.connected_components = []
    aq.connected_idxs = OrderedDict({reduce_field: {reduce_field}})
    aq.output_order = None
    aq.output_format = None
    aq.bindings = OrderedDict()
    aq.cache_point = {}

    insert_statistics(
        aq.stats_factory,
        aq.point_expr,
        aq.bindings,
        replace=False,
        cache=aq.cache_point,
    )

    for stat in aq.cache_point.values():
        dims = stat.dim_sizes
        for i in list(dims.keys()):
            if isinstance(i, str):
                dims[Field(i)] = dims[i]

    query, node_to_replace, nodes_to_remove, reduced_idxs = aq.get_reduce_query(
        reduce_field
    )

    assert query.rhs == expected


@pytest.mark.parametrize(
    ("expr", "reduce_field", "expected_query", "expected_point_expr"),
    [
        (
            # Case 1: expr = A[i] * B[j], reduce over i
            MapJoin(
                Literal(ffuncs.mul),
                (
                    Table(Literal(A), (Field("i"),)),
                    Table(Literal(B), (Field("j"),)),
                ),
            ),
            Field("i"),
            # expected query: sum_i A[i]
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (Table(Literal(A), (Field("i"),)),),
                ),
                (Field("i"),),
            ),
            # expected point expr: alias(i) * B[j]
            lambda alias_expr: MapJoin(
                Literal(ffuncs.mul),
                (
                    Table(alias_expr, ()),
                    Table(Literal(B), (Field("j"),)),
                ),
            ),
        ),
        (
            # Case 2: expr = A[i] * A[i], reduce over i
            MapJoin(
                Literal(ffuncs.mul),
                (
                    Table(Literal(A), (Field("i"),)),
                    Table(Literal(A), (Field("i"),)),
                ),
            ),
            Field("i"),
            # expected query: sum_i (A[i] * A[i])
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(A), (Field("i"),)),
                    ),
                ),
                (Field("i"),),
            ),
            # expected point expr: alias
            lambda alias_expr: Table(alias_expr, ()),
        ),
        (
            # Case 3: expr = A[i] * C[i,k] * B[j], reduce over i
            MapJoin(
                Literal(ffuncs.mul),
                (
                    Table(Literal(A), (Field("i"),)),
                    Table(Literal(C_mat), (Field("i"), Field("k"))),
                    Table(Literal(B), (Field("j"),)),
                ),
            ),
            Field("i"),
            # expected query: sum_i (A[i] * C[i,k])
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(C_mat), (Field("i"), Field("k"))),
                    ),
                ),
                (Field("i"),),
            ),
            # expected point expr: alias(k) * B[j]
            lambda alias_expr: MapJoin(
                Literal(ffuncs.mul),
                (
                    Table(alias_expr, (Field("k"),)),
                    Table(Literal(B), (Field("j"),)),
                ),
            ),
        ),
        (
            # Case 4: expr = A[i] * C[i,k] * D[k] * B[j], reduce over i
            MapJoin(
                Literal(ffuncs.mul),
                (
                    Table(Literal(A), (Field("i"),)),
                    Table(Literal(C_mat), (Field("i"), Field("k"))),
                    Table(Literal(D), (Field("k"),)),
                    Table(Literal(B), (Field("j"),)),
                ),
            ),
            Field("i"),
            # expected query: sum_i (A[i] * C[i,k] * D[k])
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(C_mat), (Field("i"), Field("k"))),
                        Table(Literal(D), (Field("k"),)),
                    ),
                ),
                (Field("i"),),
            ),
            # expected point expr: alias(k) * B[j]
            lambda alias_expr: MapJoin(
                Literal(ffuncs.mul),
                (
                    Table(alias_expr, (Field("k"),)),
                    Table(Literal(B), (Field("j"),)),
                ),
            ),
        ),
    ],
)
def test_reduce_idx(expr, reduce_field, expected_query, expected_point_expr):
    aq = object.__new__(AnnotatedQuery)
    aq.stats_factory = DenseStatsFactory()
    aq.output_name = None
    aq.reduce_idxs = [reduce_field]
    aq.point_expr = expr
    aq.idx_lowest_root = OrderedDict({reduce_field: expr})
    aq.idx_op = OrderedDict({reduce_field: ffuncs.add})
    aq.idx_init = OrderedDict({reduce_field: 0})
    aq.parent_idxs = OrderedDict()
    aq.original_idx = OrderedDict({reduce_field: reduce_field})
    aq.connected_components = []
    aq.connected_idxs = OrderedDict({reduce_field: {reduce_field}})
    aq.output_order = None
    aq.output_format = None
    aq.bindings = OrderedDict()
    aq.cache_point = {}

    insert_statistics(
        aq.stats_factory,
        aq.point_expr,
        aq.bindings,
        replace=False,
        cache=aq.cache_point,
    )

    for stat in aq.cache_point.values():
        dims = stat.dim_sizes
        for k in list(dims.keys()):
            if isinstance(k, str):
                dims[Field(k)] = dims[k]

    query = aq.reduce_idx(reduce_field)
    assert query.rhs == expected_query

    alias_expr = Alias(query.lhs.name)
    assert aq.point_expr == expected_point_expr(alias_expr)


def rename_aliases(expr):
    if isinstance(expr, Table):
        return Table(
            rename_aliases(expr.tns), tuple(rename_aliases(idx) for idx in expr.idxs)
        )
    if isinstance(expr, Alias):
        return Alias("A")
    if isinstance(expr, MapJoin):
        return MapJoin(expr.op, tuple(rename_aliases(arg) for arg in expr.args))
    if isinstance(expr, Aggregate):
        return Aggregate(
            expr.op,
            expr.init,
            expr.reduce_idxs,
            rename_aliases(expr.rhs),
        )
    return expr


@pytest.mark.parametrize(
    ("input_query", "elimination_order", "expected"),
    [
        (
            Query(
                Alias("out"),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(A), (Field("i"),)),
                    ),
                ),
            ),
            [],
            Query(
                Alias("out"),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A), (Field("i"),)),
                        Table(Literal(A), (Field("i"),)),
                    ),
                ),
            ),
        ),
        (
            Query(
                Alias("out"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    MapJoin(
                        Literal(ffuncs.mul),
                        (
                            Table(Literal(A), (Field("i"),)),
                            Table(Literal(A), (Field("j"),)),
                        ),
                    ),
                    (Field("i"),),
                ),
            ),
            [Field("i")],
            Query(
                Alias("out"),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Alias("A"), ()),
                        Table(Literal(A), (Field("j"),)),
                    ),
                ),
            ),
        ),
        (
            Query(
                Alias("out"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    MapJoin(
                        Literal(ffuncs.mul),
                        (
                            Table(Literal(A), (Field("i"),)),
                            Table(Literal(A), (Field("j"),)),
                            Table(Literal(A), (Field("k"),)),
                        ),
                    ),
                    (Field("i"), Field("j")),
                ),
            ),
            [Field("i"), Field("j")],
            # Expect: Query(out, <same MapJoin>)
            Query(
                Alias("out"),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Alias("A"), ()),
                        Table(Alias("B"), ()),
                        Table(Literal(A), (Field("k"),)),
                    ),
                ),
            ),
        ),
        (
            Query(
                Alias("out"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    Table(Literal(A), (Field("i"),)),
                    (Field("i"),),
                ),
            ),
            [
                Field("i"),
            ],
            # Expect: Query(out, <same MapJoin>)
            None,
        ),
    ],
)
def test_get_remaining_query(input_query, elimination_order, expected):
    aq = AnnotatedQuery(DenseStatsFactory(), input_query, bindings=OrderedDict())
    for field in elimination_order:
        aq.reduce_idx(field)
    query = aq.get_remaining_query()
    if expected is None:
        assert query is None
    else:
        query = Query(query.lhs, rename_aliases(query.rhs))
        expected = Query(expected.lhs, rename_aliases(expected.rhs))
        assert query == expected


@pytest.mark.parametrize(
    ("query", "reduce_field", "expected"),
    [
        (
            # Case 1: sum_{i,j,k} A[i,j] * A[j,k], reduce over i
            Query(
                Alias("out"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    MapJoin(
                        Literal(ffuncs.mul),
                        (
                            Table(Literal(A_mat), (Field("i"), Field("j"))),
                            Table(Literal(A_mat), (Field("j"), Field("k"))),
                        ),
                    ),
                    (Field("i"), Field("j"), Field("k")),
                ),
            ),
            Field("i"),
            # expected: sum_i A[i,j]
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                Table(Literal(A_mat), (Field("i"), Field("j"))),
                (Field("i"),),
            ),
        ),
        (
            # Case 2: same chain, reduce over j
            Query(
                Alias("out"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    MapJoin(
                        Literal(ffuncs.mul),
                        (
                            Table(Literal(A_mat), (Field("i"), Field("j"))),
                            Table(Literal(A_mat), (Field("j"), Field("k"))),
                        ),
                    ),
                    (Field("i"), Field("j"), Field("k")),
                ),
            ),
            Field("j"),
            # expected: unchanged full aggregate over i,j,k
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.mul),
                    (
                        Table(Literal(A_mat), (Field("i"), Field("j"))),
                        Table(Literal(A_mat), (Field("j"), Field("k"))),
                    ),
                ),
                (Field("i"), Field("j"), Field("k")),
            ),
        ),
        (
            # Case 3: same chain, reduce over k
            Query(
                Alias("out"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    MapJoin(
                        Literal(ffuncs.mul),
                        (
                            Table(Literal(A_mat), (Field("i"), Field("j"))),
                            Table(Literal(A_mat), (Field("j"), Field("k"))),
                        ),
                    ),
                    (Field("i"), Field("j"), Field("k")),
                ),
            ),
            Field("k"),
            # expected: sum_k A[j,k]
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                Table(Literal(A_mat), (Field("j"), Field("k"))),
                (Field("k"),),
            ),
        ),
        (
            # Case 4: chain_expr = sum_{i,j,k} max(A[i,j], A[j,k])
            Query(
                Alias("out"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    MapJoin(
                        Literal(ffuncs.max),
                        (
                            Table(Literal(A_mat), (Field("i"), Field("j"))),
                            Table(Literal(A_mat), (Field("j"), Field("k"))),
                        ),
                    ),
                    (Field("i"), Field("j"), Field("k")),
                ),
            ),
            Field("i"),
            # expected: unchanged
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                MapJoin(
                    Literal(ffuncs.max),
                    (
                        Table(Literal(A_mat), (Field("i"), Field("j"))),
                        Table(Literal(A_mat), (Field("j"), Field("k"))),
                    ),
                ),
                (Field("i"), Field("j"), Field("k")),
            ),
        ),
        (
            # Case 5:  sum_{j,k} max( sum_i A[i,j], A[j,k] )
            # inner Aggregate(+ over i) is already inside the MapJoin.
            Query(
                Alias("out"),
                Aggregate(
                    Literal(ffuncs.add),
                    Literal(0),
                    MapJoin(
                        Literal(ffuncs.max),
                        (
                            Aggregate(
                                Literal(ffuncs.add),
                                Literal(0),
                                Table(Literal(A_mat), (Field("i"), Field("j"))),
                                (Field("i"),),
                            ),
                            Table(Literal(A_mat), (Field("j"), Field("k"))),
                        ),
                    ),
                    (Field("j"), Field("k")),
                ),
            ),
            Field("i"),
            # expected: inner sum over i of A[i,j]
            Aggregate(
                Literal(ffuncs.add),
                Literal(0),
                Table(Literal(A_mat), (Field("i"), Field("j"))),
                (Field("i"),),
            ),
        ),
    ],
)
def test_annotated_queries(query, reduce_field, expected):
    aq = AnnotatedQuery(DenseStatsFactory(), query, bindings=OrderedDict())
    query = aq.reduce_idx(reduce_field)
    assert query.rhs == expected


def test_greedy_query_multi_component():
    """Two independent summations sum_i A[i] + sum_j B[j] produce two components."""
    fi, fj = Field("i"), Field("j")
    point_expr = MapJoin(
        Literal(ffuncs.add),
        (
            Table(Literal(A), (fi,)),
            Table(Literal(B), (fj,)),
        ),
    )
    aq = object.__new__(AnnotatedQuery)
    aq.stats_factory = DenseStatsFactory()
    aq.output_name = Alias("out")
    aq.reduce_idxs = [fi, fj]
    aq.point_expr = point_expr
    aq.idx_lowest_root = OrderedDict(
        {
            fi: Table(Literal(A), (fi,)),
            fj: Table(Literal(B), (fj,)),
        }
    )
    aq.idx_op = OrderedDict(
        {
            fi: ffuncs.add,
            fj: ffuncs.add,
        }
    )
    aq.idx_init = OrderedDict({fi: 0, fj: 0})
    aq.parent_idxs = OrderedDict({fi: [], fj: []})
    aq.original_idx = OrderedDict({fi: fi, fj: fj})
    aq.connected_components = [[fi], [fj]]
    aq.connected_idxs = OrderedDict({fi: set(), fj: set()})
    aq.output_order = None
    aq.output_format = None
    aq.bindings = OrderedDict()
    aq.cache_point = {}

    insert_statistics(
        aq.stats_factory,
        aq.point_expr,
        aq.bindings,
        replace=False,
        cache=aq.cache_point,
    )

    for stat in aq.cache_point.values():
        dims = stat.dim_sizes
        for k in list(dims.keys()):
            if isinstance(k, str):
                dims[Field(k)] = dims[k]

    # Two independent components: {i} and {j}
    assert len(aq.connected_components) == 2
    comp_names = [[f.name for f in c] for c in aq.connected_components]
    assert comp_names == [["i"], ["j"]] or comp_names == [["j"], ["i"]]

    queries = greedy_query(aq)
    # Expect: 2 reduction queries (one per index) + 1 remaining query
    assert len(queries) == 3
    # First two should be aggregates, last is the final combination
    assert isinstance(queries[0].rhs, Aggregate)
    assert isinstance(queries[1].rhs, Aggregate)
    # Component order is respected: we fully reduce first component before second
    assert queries[0].rhs.idxs in ((Field("i"),), (Field("j"),))
    assert queries[1].rhs.idxs in ((Field("i"),), (Field("j"),))
