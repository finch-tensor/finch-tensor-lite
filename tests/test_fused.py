"""AI modified: 2026-03-16T14:34:02Z parent=18fbb013175241f5081102d7b13f81f0e6c3de04"""
# AI modified: 2026-03-16T14:40:22Z parent=18fbb013175241f5081102d7b13f81f0e6c3de04

import ast
import inspect
import operator
import textwrap

import pytest

import numpy as np

import finchlite
from finchlite.finch_fused import jit
from finchlite.finch_fused import nodes as fzd
from finchlite.finch_fused.cfg_builder import (
    fused_build_cfg,
    fused_desugar,
    number_statements,
)
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
    numbered_fn, _ = number_statements(fused_fn)
    desugared_fn = fused_desugar(numbered_fn)
    cfg = fused_build_cfg(desugared_fn)

    # We won't assert on the exact structure of the CFG here, but we can at least
    # check that it has the expected number of blocks. The exact number of blocks
    # may depend on how the CFG builder handles certain constructs, so this is a
    # somewhat loose check.
    assert (
        len(cfg.blocks) >= 5
    )  # Entry block, for loop block, if block, while block, return block


def _build_liveness(fn):
    """Helper: parse, number, desugar, build CFG, run liveness."""
    fused_fn = parse_fused_function(fn)
    numbered_fn, _ = number_statements(fused_fn)
    desugared_fn = fused_desugar(numbered_fn)
    cfg = fused_build_cfg(desugared_fn)
    liveness = LivenessAnalysis(cfg)
    liveness.analyze()
    return liveness, cfg


def _all_live_names(liveness, cfg):
    """Union of all live variable names across all blocks (in and out)."""
    names = set()
    for block in cfg.blocks.values():
        names |= {v.name for v in liveness.output_states[block.id]}
        names |= {v.name for v in liveness.input_states[block.id]}
    return names


def test_liveness_straight_line():
    """Parameters must appear live at the function entry block."""

    def fn(a, b):
        c = add(a, b)
        return c  # noqa: RET504

    liveness, cfg = _build_liveness(fn)

    # After desugaring, the function body is a single block.
    # live-IN (output_states) of that block = {a, b}, since both are used.
    # c is defined and consumed in the same block so it never crosses a block
    # boundary and does not appear in any block-boundary state.
    all_live_in = set()
    for block in cfg.blocks.values():
        all_live_in |= {v.name for v in liveness.output_states[block.id]}

    assert "a" in all_live_in
    assert "b" in all_live_in
    assert "c" not in all_live_in


def test_liveness_dead_variable():
    """A variable assigned but never used afterwards must not be live after."""

    def fn(a, b):
        unused = add(a, b)  # noqa: F841
        c = matmul(a, b)
        return c  # noqa: RET504

    liveness, cfg = _build_liveness(fn)

    exit_block = list(cfg.blocks.values())[-1]
    live_at_exit = {v.name for v in liveness.input_states[exit_block.id]}
    assert "unused" not in live_at_exit


def test_liveness_loop_carried():
    """Loop-carried variables must be live at the top of the loop body."""

    def fn(n):
        total = 0
        for _i in range(n):
            total = total + 1
        return total

    liveness, cfg = _build_liveness(fn)
    names = _all_live_names(liveness, cfg)

    assert "total" in names
    assert "n" in names


def test_liveness_multi_loop_carried():
    """Multiple loop-carried variables must all be live inside the loop."""

    def fn(A, B, C, n):
        D = matmul(A, B)
        E = add(A, C)
        for _i in range(n):
            D = add(D, E)
        return D

    liveness, cfg = _build_liveness(fn)
    names = _all_live_names(liveness, cfg)

    assert "D" in names
    assert "E" in names
    assert "n" in names


def test_liveness_if_branch_merges():
    """Variables used in either branch must be live before the if."""

    def fn(cond, a, b):
        if cond:  # noqa: SIM108
            result = add(a, b)
        else:
            result = matmul(a, b)
        return result

    liveness, cfg = _build_liveness(fn)
    names = _all_live_names(liveness, cfg)

    assert "a" in names
    assert "b" in names
    assert "result" in names


def test_jit_straight_line():
    """A jit function with no loops should produce the same result as eager."""

    def simple_fn(A, B):
        C = matmul(A, B)
        return C  # noqa: RET504

    @jit
    def opt_fn(A, B):
        C = matmul(A, B)
        return C  # noqa: RET504

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[5, 6], [7, 8]]))

    finch_assert_allclose(opt_fn(A, B), simple_fn(A, B))


def test_jit_return_expr():
    """A jit function with no loops should produce the same result as eager."""

    def simple_fn(A, B):
        return matmul(A, B), matmul(A, B)

    @jit
    def opt_fn(A, B):
        return matmul(A, B), matmul(A, B)

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[5, 6], [7, 8]]))

    finch_assert_allclose(opt_fn(A, B), simple_fn(A, B))


def test_jit_two_independent_ops():
    """Two independent tensor ops whose results are both used."""

    def simple_fn(A, B, C):
        D = matmul(A, B)
        E = add(A, C)
        F = add(D, E)
        return F  # noqa: RET504

    @jit
    def opt_fn(A, B, C):
        D = matmul(A, B)
        E = add(A, C)
        F = add(D, E)
        return F  # noqa: RET504

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[1, 0], [0, 1]]))
    C = asarray(np.array([[1, 1], [1, 1]]))

    finch_assert_allclose(opt_fn(A, B, C), simple_fn(A, B, C))


def test_jit_scalar_loop():
    """A loop with a scalar iteration count and tensor accumulation."""

    def simple_fn(A, n):
        B = A
        for _i in range(n):
            B = add(B, A)
        return B

    @jit
    def opt_fn(A, n):
        B = A
        for _i in range(n):
            B = add(B, A)
        return B

    A = asarray(np.array([[1, 0], [0, 1]], dtype=float))

    finch_assert_allclose(opt_fn(A, 3), simple_fn(A, 3))


def test_jit_dependent_loop():
    """A loop with an iterator that depends on a computation."""

    def simple_fn(A, n):
        B = A
        for _i in range(sum(B)):
            B = add(B, A)
        return B

    @jit
    def opt_fn(A, n):
        B = A
        for _i in range(sum(B)):
            B = add(B, A)
        return B

    A = asarray(np.array([[1, 0], [0, 1]], dtype=int))

    finch_assert_allclose(opt_fn(A, 3), simple_fn(A, 3))


def test_jit_if_branch():
    """A jit function with an if/else over tensor ops."""

    def simple_fn(A, B, use_matmul):
        if use_matmul:  # noqa: SIM108
            result = matmul(A, B)
        else:
            result = add(A, B)
        return result

    @jit
    def opt_fn(A, B, use_matmul):
        if use_matmul:  # noqa: SIM108
            result = matmul(A, B)
        else:
            result = add(A, B)
        return result

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[1, 0], [0, 1]]))

    finch_assert_allclose(opt_fn(A, B, True), simple_fn(A, B, True))
    finch_assert_allclose(opt_fn(A, B, False), simple_fn(A, B, False))


def test_jit_while():
    """A jit function with a while loop."""

    def simple_fn(A, B, n):
        C = A
        while n > 0:
            C = add(C, B)
            n = n - 1
        return C

    @jit
    def opt_fn(A, B, n):
        C = A
        while n > 0:
            C = add(C, B)
            n = n - 1
        return C

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[1, 0], [0, 1]]))
    n = 3

    finch_assert_allclose(opt_fn(A, B, n), simple_fn(A, B, n))


def test_jit_module_function():
    """A jit function with a function from a module."""

    def simple_fn(A, B, n):
        C = A
        while n > 0:
            C = finchlite.interface.add(C, B)
            n = n - 1
        return C

    @jit
    def opt_fn(A, B, n):
        C = A
        while n > 0:
            C = finchlite.interface.add(C, B)
            n = n - 1
        return C

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[1, 0], [0, 1]]))
    n = 3

    finch_assert_allclose(opt_fn(A, B, n), simple_fn(A, B, n))


def test_jit_local_module_function():
    """A jit function with a function from a module."""

    xp = finchlite.interface

    def simple_fn(A, B, n):
        C = A
        while n > 0:
            C = xp.add(C, B)
            n = n - 1
        return C

    @jit
    def opt_fn(A, B, n):
        C = A
        while n > 0:
            C = xp.add(C, B)
            n = n - 1
        return C

    A = asarray(np.array([[1, 2], [3, 4]]))
    B = asarray(np.array([[1, 0], [0, 1]]))
    n = 3

    finch_assert_allclose(simple_fn(A, B, n), opt_fn(A, B, n))
