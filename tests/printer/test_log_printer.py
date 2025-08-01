import operator
from textwrap import dedent

import numpy as np

import finch.finch_logic as log
from finch.finch_logic.printer import PrinterCompiler


def test_printer():
    pc = PrinterCompiler()

    s = np.array([[2, 4], [6, 0]])
    a = np.array([[1, 2], [3, 2]])
    b = np.array([[9, 8], [6, 5]])
    i, j, k = log.Field("i"), log.Field("j"), log.Field("k")

    prgm = log.Plan(
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

    actual = pc(prgm)

    expected = dedent("""\
        S = Table([[2 4] [6 0]], ['i', 'j'])
        A = Table([[1 2] [3 2]], ['i', 'k'])
        B = Table([[9 8] [6 5]], ['k', 'j'])
        AB = MapJoin(mul, ('A', 'B'))
        C = Aggregate(add, 0, AB, ['k'])
        RES = MapJoin(mul, ('C', 'S'))
        return ('RES',)
        """)

    assert expected == actual
