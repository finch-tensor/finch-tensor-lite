from finchlite.finch_logic import Field, Literal, MapJoin, Plan, Produces, Table
from finchlite.symbolic import PostOrderDFS, PreOrderDFS, intree, isdescendant

def test_preorder_logic():
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
    type_names = [type(x).__name__ for x in list(PreOrderDFS(prog))]

    assert type_names == [
        "Plan",
        "Produces",
        "MapJoin",
        "Field",
        "Table",
        "Literal",
        "Field",
        "Field",
        "Table",
        "Literal",
        "Field",
        "Field",
    ]


def test_postorder_logic():
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

    post = list(PostOrderDFS(prog))
    type_names = [type(x).__name__ for x in post]

    assert type_names == [
        "Field",
        "Literal",
        "Field",
        "Field",
        "Table",
        "Literal",
        "Field",
        "Field",
        "Table",
        "MapJoin",
        "Produces",
        "Plan",
    ]

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
