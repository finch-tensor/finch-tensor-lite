from collections import Counter

from finchlite.finch_logic import Field, Literal, MapJoin, Plan, Produces, Table
from finchlite.symbolic import PostOrderDFS, PreOrderDFS, intree, isdescendant


def test_preorder_dfs():
    ta = Table(
        Literal("A"),
        (Field("i"), Field("j")),
    )

    tb = Table(
        Literal("B"),
        (Field("j"), Field("k")),
    )

    prog = Plan(
        (
            Produces(
                (
                    MapJoin(
                        Field("op"),
                        (ta, tb),
                    ),
                ),
            ),
        )
    )

    preorder = list(PreOrderDFS(prog))

    assert Counter(type(x).__name__ for x in preorder) == Counter(
        {"Plan": 1, "Produces": 1, "MapJoin": 1, "Table": 2, "Literal": 2, "Field": 5}
    )

    for i, node in enumerate(preorder):
        for child in getattr(node, "children", ()):
            child_pos = next(
                (j for j, n in enumerate(preorder) if n is child and j > i), None
            )
            assert child_pos is not None


def test_postorder_dfs():
    ta = Table(
        Literal("A"),
        (Field("i"), Field("j")),
    )

    tb = Table(
        Literal("B"),
        (Field("j"), Field("k")),
    )

    prog = Plan(
        (
            Produces(
                (
                    MapJoin(
                        Field("op"),
                        (ta, tb),
                    ),
                ),
            ),
        )
    )

    postorder = list(PostOrderDFS(prog))

    assert Counter(type(x).__name__ for x in postorder) == Counter(
        {"Plan": 1, "Produces": 1, "MapJoin": 1, "Table": 2, "Literal": 2, "Field": 5}
    )

    for i, node in enumerate(postorder):
        for child in getattr(node, "children", ()):
            child_pos = next(
                (j for j, n in enumerate(postorder) if n is child and j < i), None
            )
            assert child_pos is not None


def test_intree():
    i, j, k = Field("i"), Field("j"), Field("k")
    ta = Table(Literal("A"), (i, j))
    tb = Table(Literal("B"), (j, k))
    op = Field("op")
    mj = MapJoin(op, (ta, tb))
    prog = Plan((Produces((mj,)),))

    assert intree(prog, prog)
    assert intree(mj, prog)
    assert intree(ta, prog)
    assert intree(tb, prog)


def test_isdescendant():
    i, j, k = Field("i"), Field("j"), Field("k")
    ta = Table(Literal("A"), (i, j))
    tb = Table(Literal("B"), (j, k))
    op = Field("op")
    mj = MapJoin(op, (ta, tb))
    prog = Plan((Produces((mj,)),))

    assert isdescendant(mj, prog)
    assert isdescendant(ta, prog)
    assert isdescendant(tb, prog)
