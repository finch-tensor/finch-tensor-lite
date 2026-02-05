import operator
from typing import Any

import numpy as np

from lark import Lark, Token, Tree

from ..algebra import overwrite, promote_max, promote_min
from ..symbolic import Namespace
from . import nodes as ein

nary_ops = {
    "+": operator.add,
    "add": operator.add,
    "-": operator.sub,
    "sub": operator.sub,
    "subtract": operator.sub,
    "*": operator.mul,
    "mul": operator.mul,
    "multiply": operator.mul,
    "/": operator.truediv,
    "div": operator.truediv,
    "divide": operator.truediv,
    "//": operator.floordiv,
    "fld": operator.floordiv,
    "floor_divide": operator.floordiv,
    "%": operator.mod,
    "mod": operator.mod,
    "remainder": operator.mod,
    "**": operator.pow,
    "pow": operator.pow,
    "power": operator.pow,
    "==": operator.eq,
    "eq": operator.eq,
    "equal": operator.eq,
    "!=": operator.ne,
    "ne": operator.ne,
    "not_equal": operator.ne,
    "<": operator.lt,
    "lt": operator.lt,
    "less": operator.lt,
    "<=": operator.le,
    "le": operator.le,
    "less_equal": operator.le,
    ">": operator.gt,
    "gt": operator.gt,
    "greater": operator.gt,
    ">=": operator.ge,
    "ge": operator.ge,
    "greater_equal": operator.ge,
    "&": operator.and_,
    "bitwise_and": operator.and_,
    "|": operator.or_,
    "bitwise_or": operator.or_,
    "^": operator.xor,
    "bitwise_xor": operator.xor,
    "<<": operator.lshift,
    "lshift": operator.lshift,
    "bitwise_left_shift": operator.lshift,
    ">>": operator.rshift,
    "rshift": operator.rshift,
    "bitwise_right_shift": operator.rshift,
    "and": np.logical_and,
    "or": np.logical_or,
    "not": np.logical_not,
    "min": promote_min,
    "max": promote_max,
    "logaddexp": np.logaddexp,
}


unary_ops = {
    "+": operator.pos,
    "pos": operator.pos,
    "positive": operator.pos,
    "-": operator.neg,
    "neg": operator.neg,
    "negative": operator.neg,
    "~": operator.invert,
    "invert": operator.invert,
    "bitwise_invert": operator.invert,
    "not": np.logical_not,
    "logical_not": np.logical_not,
    "abs": operator.abs,
    "absolute": operator.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log1p": np.log1p,
    "log10": np.log10,
    "log2": np.log2,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asin": np.arcsin,
    "arcsin": np.arcsin,
    "acos": np.arccos,
    "arccos": np.arccos,
    "atan": np.arctan,
    "arctan": np.arctan,
    "asinh": np.arcsinh,
    "arcsinh": np.arcsinh,
    "acosh": np.arccosh,
    "arccosh": np.arccosh,
    "atanh": np.arctanh,
    "arctanh": np.arctanh,
}


reduction_ops = {
    "+": operator.add,
    "add": operator.add,
    "sum": operator.add,
    "*": operator.mul,
    "mul": operator.mul,
    "prod": operator.mul,
    "and": np.logical_and,
    "all": np.logical_and,
    "or": np.logical_or,
    "any": np.logical_or,
    "min": promote_min,
    "minimum": promote_min,
    "max": promote_max,
    "maximum": promote_max,
    "&": operator.and_,
    "bitwise_and": operator.and_,
    "|": operator.or_,
    "bitwise_or": operator.or_,
    "^": operator.xor,
    "bitwise_xor": operator.xor,
}


lark_parser = Lark("""
    %import common.CNAME
    %import common.SIGNED_INT
    %import common.SIGNED_FLOAT
    %ignore " "           // Disregard spaces in text

    start: increment | assign
    increment: access (OP | FUNC_NAME) "=" expr
    assign: access "=" expr

    // Python operator precedence (lowest to highest)
    expr: or_expr
    or_expr: and_expr (OR and_expr)*
    and_expr: not_expr (AND not_expr)*
    not_expr: NOT not_expr | comparison_expr
    comparison_expr: bitwise_or_expr ((EQ | NE | LT | LE | GT | GE) bitwise_or_expr)*
    bitwise_or_expr: bitwise_xor_expr (PIPE bitwise_xor_expr)*
    bitwise_xor_expr: bitwise_and_expr (CARET bitwise_and_expr)*
    bitwise_and_expr: shift_expr (AMPERSAND shift_expr)*
    shift_expr: add_expr ((LSHIFT | RSHIFT) add_expr)*
    add_expr: mul_expr ((PLUS | MINUS) mul_expr)*
    mul_expr: unary_expr ((MUL | DIV | FLOORDIV | MOD) unary_expr)*
    unary_expr: (PLUS | MINUS | TILDE) unary_expr | power_expr
    power_expr: primary (POW unary_expr)?
    primary: call_func | access | literal | IDX | "(" expr ")"

    OR: "or"
    AND: "and"
    NOT: "not"
    EQ: "=="
    NE: "!="
    LT: "<"
    LE: "<="
    GT: ">"
    GE: ">="
    PIPE: "|"
    CARET: "^"
    AMPERSAND: "&"
    LSHIFT: "<<"
    RSHIFT: ">>"
    PLUS: "+"
    MINUS: "-"
    MUL: "*"
    DIV: "/"
    FLOORDIV: "//"
    MOD: "%"
    POW: "**"
    TILDE: "~"

    OP: "+" | "-" | "*" | "or" | "and" | "|" | "&" | "^" | "<<" | ">>"
          | "//" | "/" | "%" | "**" | ">" | "<" | ">=" | "<=" | "==" | "!="

    // Tensor expression: supports attribute access
    // e.g., A, A.elems, A.coords[0][i]
    tns_expr: tns_getattr | tns_base
    tns_getattr: tns_expr "." ATTR
    tns_base: TNS

    // Top-level access used in assignments (Result[i] = ...)
    access: tns_expr "[" (primary ",")* primary? "]"

    call_func: (FUNC_NAME "(" (expr ",")* expr?  ")")
    literal: bool_literal | complex_literal | float_literal | int_literal
    bool_literal: BOOL
    int_literal: SIGNED_INT
    float_literal: SIGNED_FLOAT
    complex_literal: COMPLEX

    BOOL: "True" | "False"
    COMPLEX: (SIGNED_FLOAT | SIGNED_INT) ("j" | "J")
    IDX: CNAME
    TNS: CNAME
    ATTR: CNAME
    FUNC_NAME: CNAME
""")


def _parse_einop_expr(t: Tree | Token) -> ein.EinsumExpression:
    match t:
        case Tree(
            "start"
            | "expr"
            | "or_expr"
            | "and_expr"
            | "not_expr"
            | "comparison_expr"
            | "bitwise_or_expr"
            | "bitwise_xor_expr"
            | "bitwise_and_expr"
            | "shift_expr"
            | "add_expr"
            | "mul_expr"
            | "unary_expr"
            | "power_expr"
            | "primary"
            | "literal",
            [child],
        ):
            # Handle IDX tokens directly to avoid recursing with a non-Tree
            if isinstance(child, Token) and child.type == "IDX":
                return ein.Index(str(child.value))
            return _parse_einop_expr(child)
        case Tree(
            "or_expr"
            | "and_expr"
            | "bitwise_or_expr"
            | "bitwise_and_expr"
            | "bitwise_xor_expr"
            | "shift_expr"
            | "add_expr"
            | "mul_expr",
            args,
        ) if len(args) > 1:
            expr = _parse_einop_expr(args[0])
            for i in range(1, len(args), 2):
                arg = _parse_einop_expr(args[i + 1])
                op = ein.Literal(nary_ops[args[i].value])  # type: ignore[union-attr]
                expr = ein.Call(op, (expr, arg))
            return expr
        case Tree("comparison_expr", args) if len(args) > 1:
            # Handle Python's comparison chaining: a < b < c becomes (a < b) and (b < c)
            left = _parse_einop_expr(args[0])
            right = _parse_einop_expr(args[2])
            op = ein.Literal(nary_ops[args[1].value])  # type: ignore[union-attr]
            expr = ein.Call(op, (left, right))
            for i in range(2, len(args) - 2, 2):
                left = _parse_einop_expr(args[i])
                right = _parse_einop_expr(args[i + 2])
                and_ = ein.Literal(nary_ops["and"])  # type: ignore[union-attr]
                op = ein.Literal(nary_ops[args[i + 1].value])  # type: ignore[union-attr]
                expr = ein.Call(and_, (expr, ein.Call(op, (left, right))))  # type: ignore[union-attr]
            return expr
        case Tree("power_expr", args) if len(args) > 1:
            left = _parse_einop_expr(args[0])
            right = _parse_einop_expr(args[2])
            op = ein.Literal(nary_ops[args[1].value])  # type: ignore[union-attr]
            return ein.Call(op, (left, right))
        case Tree("unary_expr" | "not_expr", [op, arg]):
            op = ein.Literal(unary_ops[op.value])  # type: ignore[union-attr]
            return ein.Call(op, (_parse_einop_expr(arg),))
        case Tree("access", [tns_expr, *idxs]):
            return ein.Access(
                _parse_einop_tns(tns_expr),
                tuple(_parse_einop_expr(idx) for idx in idxs),
            )
        case Tree("bool_literal", (val,)):
            return ein.Literal(val.value == "True")  # type: ignore[union-attr]
        case Tree("int_literal", (val,)):
            return ein.Literal(int(val.value))  # type: ignore[union-attr]
        case Tree("float_literal", (val,)):
            return ein.Literal(float(val.value))  # type: ignore[union-attr]
        case Tree("complex_literal", (val,)):
            return ein.Literal(complex(val.value))  # type: ignore[union-attr]
        case Tree("call_func", [func, *args]):
            return ein.Call(func.value, (*(_parse_einop_expr(arg) for arg in args),))  # type: ignore[union-attr]
        case _:
            raise ValueError(f"Unknown tree structure: {t}")


def _parse_einop_tns(tns_expr: Tree | Token) -> str:
    """Extract the tensor name from a tns_expr tree (expecting tns_base)."""
    match tns_expr:
        case Tree("tns_expr", [child]):
            return _parse_einop_tns(child)
        case Tree("tns_getattr", [obj, attr]):
            return ein.GetAttr(
                obj=_parse_einop_tns(obj),
                attr=ein.Literal(attr.value),  # type: ignore[union-attr]
            )
        case Tree("tns_base", [tns]):
            return ein.Alias(tns.value)  # type: ignore[union-attr]
        case _:
            raise ValueError(f"Unknown tree structure: {tns_expr}")


def parse_einop(expr: str) -> ein.EinsumNode:
    tree = lark_parser.parse(expr)
    match tree:
        case Tree(
            "start",
            [
                Tree(
                    "increment",
                    [Tree("access", [tns_expr, *idxs]), op_token, expr_node],
                )
            ],
        ):
            arg = _parse_einop_expr(expr_node)  # type: ignore[arg-type]
            op = ein.Literal(reduction_ops[op_token.value])  # type: ignore[union-attr]
            tns = _parse_einop_tns(tns_expr)
            if not isinstance(tns, ein.Alias):
                raise ValueError(f"Increment target must be a tensor alias, got {tns}.")
            return ein.Einsum(
                op,
                tns,
                tuple(_parse_einop_expr(idx) for idx in idxs),
                arg,  # type: ignore[union-attr]
            )

        case Tree(
            "start", [Tree("assign", [Tree("access", [tns_expr, *idxs]), expr_node])]
        ):
            arg = _parse_einop_expr(expr_node)  # type: ignore[arg-type]
            op = ein.Literal(overwrite)
            tns = _parse_einop_tns(tns_expr)
            if not isinstance(tns, ein.Alias):
                raise ValueError(f"Increment target must be a tensor alias, got {tns}.")
            return ein.Einsum(
                op,
                tns,
                tuple(_parse_einop_expr(idx) for idx in idxs),
                arg,
            )

        case _:
            raise ValueError(
                f"Expected top-level assignment or increment, got {tree.data}"
            )


def parse_einsum(*args_) -> tuple[ein.EinsumNode, dict[ein.Alias, Any]]:
    args = list(args_)
    if len(args) < 2:
        raise ValueError("Expected at least a subscript string and one operand.")
    bc = "none"
    if isinstance(args[0], str):
        subscripts = args[0]
        operands = args[1:]
        if subscripts.count("->") > 1:
            raise ValueError("Subscripts can only contain one '->' symbol.")
        if subscripts.count("->") == 1:
            subscripts, output_sub = subscripts.split("->")
            output_sub = output_sub.strip()
        else:
            output_sub = None
        input_subs = [s.strip() for s in subscripts.split(",")]
        # Check for ellipses in input subscripts
        if any("..." in sub for sub in input_subs):
            if all(sub.startswith("...") for sub in input_subs):
                bc = "prefix"
                input_subs = [sub[3:] for sub in input_subs]
                if output_sub is not None:
                    assert output_sub.startswith("...")
                    output_sub = output_sub[3:]
            elif all(sub.endswith("...") for sub in input_subs):
                bc = "suffix"
                input_subs = [sub[:-3] for sub in input_subs]
                if output_sub is not None:
                    assert output_sub.endswith("...")
                    output_sub = output_sub[:-3]
            else:
                raise ValueError(
                    "Ellipses must be at the start or end of all subscripts."
                )
        input_idxs = [list(sub) for sub in input_subs]
        output_idxs = None if output_sub is None else list(output_sub)
    else:
        # Alternative syntax: einsum(operand0, subscript0, operand1, subscript1, ...)
        # Check if the last element is the output subscript
        if len(args) % 2 == 1:
            operands = args[0:-2:2]
            input_idxs = args[1::2]
            output_idxs = list(args[-1])
            output_idxs = [f"j_{j}" for j in output_idxs]
        else:
            operands = args[0::2]
            input_idxs = args[1::2]
            output_idxs = None
        input_idxs = [[f"j_{j}" for j in idx] for idx in input_idxs]
        if any(Ellipsis in idx for idx in input_idxs):
            if all(idx[0] == Ellipsis for idx in input_idxs):
                bc = "prefix"
                input_idxs = [idx[1:] for idx in input_idxs]
                if output_idxs is not None:
                    assert output_idxs[0] == Ellipsis
                    output_idxs = output_idxs[1:]
            elif all(idx[-1] == Ellipsis for idx in input_idxs):
                bc = "suffix"
                input_idxs = [idx[:-1] for idx in input_idxs]
                if output_idxs is not None:
                    assert output_idxs[-1] == Ellipsis
                    output_idxs = output_idxs[:-1]
            else:
                raise ValueError(
                    "Ellipses must be at the start or end of all subscripts."
                )

    all_idxs = set().union(*input_idxs)

    if output_idxs is None:
        output_idx_set = set()
        for idx in all_idxs:
            if sum(idx in sub for sub in input_idxs) == 1:
                output_idx_set.add(idx)
        output_idxs = sorted(output_idx_set)

    def ndim(tns):
        if hasattr(tns, "ndim"):
            return tns.ndim
        return 0

    if bc == "prefix":
        max_ell_len = max(
            ndim(op) - len(sub) for op, sub in zip(operands, input_idxs, strict=False)
        )
        for i in range(len(operands)):
            ell_idxs = [
                f"i_{j}"
                for j in range(
                    max_ell_len - (ndim(operands[i]) - len(input_idxs[i])), max_ell_len
                )
            ]
            input_idxs[i] = ell_idxs + input_idxs[i]
        ell_idxs = [f"i_{j}" for j in range(max_ell_len)]
        output_idxs = [f"i_{j}" for j in range(max_ell_len)] + output_idxs
    elif bc == "suffix":
        max_ell_len = max(
            ndim(op) - len(sub) for op, sub in zip(operands, input_idxs, strict=False)
        )
        for i in range(len(operands)):
            ell_idxs = [f"k_{j}" for j in range(ndim(operands[i]) - len(input_idxs[i]))]
            input_idxs[i] = input_idxs[i] + ell_idxs
        output_idxs = output_idxs + [f"k_{j}" for j in range(max_ell_len)]

    all_idxs = set().union(*input_idxs)

    if len(input_idxs) != len(operands):
        raise ValueError("Number of input subscripts must match number of operands.")
    assert set(output_idxs).issubset(all_idxs), (
        "Output indices must be a subset of input indices."
    )
    spc = Namespace()
    for j in all_idxs:
        spc.freshen(j)
    if output_idxs == all_idxs:
        op = ein.Literal(overwrite)
    else:
        op = ein.Literal(operator.add)
    out_tns = ein.Alias(spc.freshen("B"))
    idxs = tuple(ein.Index(j) for j in output_idxs)
    in_tnss = [ein.Alias(spc.freshen("A")) for _ in operands]
    arg = ein.Access(in_tnss[0], tuple(ein.Index(i) for i in input_idxs[0]))
    for i in range(1, len(operands)):
        arg = ein.Call(
            ein.Literal(operator.mul),
            (arg, ein.Access(in_tnss[i], tuple(ein.Index(j) for j in input_idxs[i]))),
        )  # type: ignore[assignment]
    return (
        ein.Einsum(
            op,
            out_tns,
            idxs,
            arg,
        ),
        {in_tnss[i]: operands[i] for i in range(len(operands))},
    )
