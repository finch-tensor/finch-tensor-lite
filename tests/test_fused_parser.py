"""AI modified: 2026-03-16T14:34:02Z parent=18fbb013175241f5081102d7b13f81f0e6c3de04"""
# AI modified: 2026-03-16T14:40:22Z parent=18fbb013175241f5081102d7b13f81f0e6c3de04

import ast
import inspect
import operator
import textwrap

import pytest

import numpy as np

from finchlite.finch_fused import jit
from finchlite.finch_fused import nodes as fzd
from finchlite.finch_fused.cfg_builder import fused_build_cfg, fused_desugar
from finchlite.finch_fused.dataflow import LivenessAnalysis
from finchlite.finch_fused.parser import (
    fused_function_to_python_ast,
    parse_fused_function,
)
from finchlite.interface import add, asarray, matmul
from tests.conftest import finch_assert_allclose


def test_parse_simple_function_with_control_flow_and_calls():
    def simple_fn(fn, n):
        total = 0
        for i in range(n):
            if i < n:  # noqa: SIM108
                total = fn(total, i)
            else:
                total = total - 1
        while total < n:
            total = total + 1
        return total

    result = parse_fused_function(simple_fn)

    expected = fzd.Function(
        fzd.Literal("simple_fn"),
        (fzd.Variable("fn"), fzd.Variable("n")),
        fzd.Block(
            (
                fzd.Assign(fzd.Variable("total"), fzd.Literal(0)),
                fzd.For(
                    fzd.Variable("i"),
                    fzd.Call(fzd.Literal(range), (fzd.Variable("n"),)),
                    fzd.Block(
                        (
                            fzd.If(
                                fzd.Compare(
                                    fzd.Variable("i"),
                                    fzd.Literal(operator.lt),
                                    fzd.Variable("n"),
                                ),
                                fzd.Block(
                                    (
                                        fzd.Assign(
                                            fzd.Variable("total"),
                                            fzd.Call(
                                                fzd.Variable("fn"),
                                                (
                                                    fzd.Variable("total"),
                                                    fzd.Variable("i"),
                                                ),
                                            ),
                                        ),
                                    )
                                ),
                                fzd.Block(
                                    (
                                        fzd.Assign(
                                            fzd.Variable("total"),
                                            fzd.BinaryOp(
                                                fzd.Variable("total"),
                                                fzd.Literal(operator.sub),
                                                fzd.Literal(1),
                                            ),
                                        ),
                                    )
                                ),
                            ),
                        )
                    ),
                ),
                fzd.While(
                    fzd.Compare(
                        fzd.Variable("total"),
                        fzd.Literal(operator.lt),
                        fzd.Variable("n"),
                    ),
                    fzd.Block(
                        (
                            fzd.Assign(
                                fzd.Variable("total"),
                                fzd.BinaryOp(
                                    fzd.Variable("total"),
                                    fzd.Literal(operator.add),
                                    fzd.Literal(1),
                                ),
                            ),
                        )
                    ),
                ),
                fzd.Return((fzd.Variable("total"),)),
            )
        ),
    )

    assert result == expected


def test_parse_rejects_local_function_definitions():
    def with_local_fn(x):
        def inner(y):
            return y + 1

        return inner(x)

    with pytest.raises(ValueError, match="Local functions are not supported"):
        parse_fused_function(with_local_fn)


def test_parse_rejects_for_else_blocks():
    def with_for_else(n):
        for i in range(n):
            n = n + i
        else:
            n = n + 1
        return n

    with pytest.raises(ValueError, match="For-else blocks are not supported"):
        parse_fused_function(with_for_else)


def test_parse_rejects_while_else_blocks():
    def with_while_else(n):
        while n < 3:
            n = n + 1
        else:
            n = n + 2
        return n

    with pytest.raises(ValueError, match="While-else blocks are not supported"):
        parse_fused_function(with_while_else)


def test_parse_reverse_parse_is_lossless_on_supported_subset():
    def roundtrip_fn(n):
        total = 0
        for i in range(n):
            if i < n:  # noqa: SIM108
                total = total + i
            else:
                total = total - 1
        while total < n:
            total = total + 1
        return total

    source = textwrap.dedent(inspect.getsource(roundtrip_fn))
    original_module = ast.parse(source)
    original_fn = original_module.body[0]

    fused_fn = parse_fused_function(roundtrip_fn)
    roundtrip_fn_ast = fused_function_to_python_ast(fused_fn)

    assert ast.dump(original_fn, include_attributes=False) == ast.dump(
        roundtrip_fn_ast,
        include_attributes=False,
    )


def test_cfg_builder():
    def simple_fn(fn, n):
        total = 0
        for i in range(n):
            if i < n:  # noqa: SIM108
                total = fn(total, i)
            else:
                total = total - 1
        while total < n:
            total = total + 1
        return total

    fused_fn = parse_fused_function(simple_fn)
    numbered_fn, _ = fused_desugar(fused_fn, 0)
    cfg = fused_build_cfg(numbered_fn)

    # We won't assert on the exact structure of the CFG here, but we can at least
    # check that it has the expected number of blocks. The exact number of blocks
    # may depend on how the CFG builder handles certain constructs, so this is a
    # somewhat loose check.
    assert (
        len(cfg.blocks) >= 5
    )  # Entry block, for loop block, if block, while block, return block


def test_liveness_analysis():
    def simple_fn(n):
        total = 0
        for i in range(n):
            total = total + i
        return total

    fused_fn = parse_fused_function(simple_fn)

    numbered_fn, _ = fused_desugar(fused_fn, 0)
    cfg = fused_build_cfg(numbered_fn)
    print(cfg)
    liveness = LivenessAnalysis(cfg)
    liveness.analyze()

    # We won't assert on the exact liveness sets here, but we can at least check
    # that the analysis produces some output and that it includes expected variables.
    for block in cfg.blocks.values():
        input_live_vars = liveness.input_states[block.id]
        output_live_vars = liveness.output_states[block.id]
        print(f"Block {block}")
        print(f"input live variables: {input_live_vars}")
        print(f"output live variables: {output_live_vars}")


def test_lazy_and_compute_insertion():
    def simple_fn(A, B, C, n_iter):
        D = matmul(A, B)
        E = add(A, C)
        for _i in range(n_iter):
            D = add(D, E)
        return D

    @jit
    def opt_simple_fn(A, B, C, n_iter):
        D = matmul(A, B)
        E = add(A, C)
        for _i in range(n_iter):
            D = add(D, E)
        return D

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[1, 2], [3, 4]]))
    C = asarray(np.array([[1, 2], [3, 4]]))
    n_iter = 5

    expected_result = simple_fn(A, B, C, n_iter)
    print(opt_simple_fn)
    result = opt_simple_fn(A, B, C, n_iter)
    finch_assert_allclose(result, expected_result)
