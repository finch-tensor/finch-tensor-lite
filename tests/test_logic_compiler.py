from finch.autoschedule import (
    LogicCompiler,
)


def test_logic_compiler():
    i, j, k = Field("i"), Field("j"), Field("k")

    plan = Plan(
        (
            Query(Alias("A"), Table(Literal(a), (i, k))),
            Query(Alias("B"), Table(Literal(b), (k, j))),
            Query(Alias("AB"), MapJoin(Literal(mul), (Alias("A"), Alias("B")))),
            Query(
                Alias("C"),
                Reorder(Aggregate(Literal(add), Literal(0), Alias("AB"), (k,)), (i, j)),
            ),
            Produces((Alias("C"),)),
        )
    )

    prgm = LogicCompiler()(plan_opt)
