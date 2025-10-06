import operator
import builtins

import numpy as np

from lark import Lark, Tree

from . import nodes as ein
from ..algebra import overwrite

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
    "min": builtins.min,
    "max": builtins.max,
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
    "min": builtins.min,
    "minimum": builtins.min,
    "max": builtins.max,
    "maximum": builtins.max,
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
    primary: call_func | access | literal | "(" expr ")"

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

    access: TNS "[" (IDX ",")* IDX? "]"
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
    FUNC_NAME: CNAME
""")


def _parse_einsum_expr(t: Tree) -> ein.EinsumExpr:
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
            return _parse_einsum_expr(child)
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
            expr = _parse_einsum_expr(args[0])
            for i in range(1, len(args), 2):
                arg = _parse_einsum_expr(args[i + 1])
                op = ein.Literal(nary_ops[args[i].value])  # type: ignore[union-attr]
                expr = ein.Call(op, (expr, arg))
            return expr
        case Tree("comparison_expr", args) if len(args) > 1:
            # Handle Python's comparison chaining: a < b < c becomes (a < b) and (b < c)
            left = _parse_einsum_expr(args[0])
            right = _parse_einsum_expr(args[2])
            op = ein.Literal(nary_ops[args[1].value])  # type: ignore[union-attr]
            expr = ein.Call(op, (left, right))
            for i in range(2, len(args) - 2, 2):
                left = _parse_einsum_expr(args[i])
                right = _parse_einsum_expr(args[i + 2])
                and_ = ein.Literal(nary_ops["and"])  # type: ignore[union-attr]
                op = ein.Literal(args[i + 1].value)  # type: ignore[union-attr]
                expr = ein.Call(
                    and_, (expr, ein.Call(op, (left, right)))
                )  # type: ignore[union-attr]
            return expr
        case Tree("power_expr", args) if len(args) > 1:
            left = _parse_einsum_expr(args[0])
            right = _parse_einsum_expr(args[2])
            op = ein.Literal(nary_ops[args[1].value])  # type: ignore[union-attr]
            return ein.Call(op, (left, right))
        case Tree("unary_expr" | "not_expr", [op, arg]):
            op = ein.Literal(unary_ops[op.value])  # type: ignore[union-attr]
            return ein.Call(op, (_parse_einsum_expr(arg),))
        case Tree("access", [tns, *idxs]):
            return ein.Access(ein.Alias(tns.value), (*(ein.Index(idx.value) for idx in idxs),))  # type: ignore[union-attr]
        case Tree("bool_literal", (val,)):
            return ein.Literal(val.value == "True")  # type: ignore[union-attr]
        case Tree("int_literal", (val,)):
            return ein.Literal(int(val.value))  # type: ignore[union-attr]
        case Tree("float_literal", (val,)):
            return ein.Literal(float(val.value))  # type: ignore[union-attr]
        case Tree("complex_literal", (val,)):
            return ein.Literal(complex(val.value))  # type: ignore[union-attr]
        case Tree("call_func", [func, *args]):
            return ein.Call(func.value, (*(_parse_einsum_expr(arg) for arg in args),))  # type: ignore[union-attr]
        case _:
            raise ValueError(f"Unknown tree structure: {t}")


def parse_einsum(expr: str) -> ein.EinsumNode:
    tree = lark_parser.parse(expr)
    match tree:
        case Tree(
            "start", [Tree("increment", [Tree("access", [tns, *idxs]), op, expr_node])]
        ):
            arg = _parse_einsum_expr(expr_node)  # type: ignore[arg-type]
            idxs_exprs = tuple(ein.Index(idx.value) for idx in idxs)  # type: ignore[union-attr]
            return ein.Einsum(
                op.value, ein.Alias(tns.value), idxs_exprs, arg   # type: ignore[union-attr]
            )

        case Tree("start", [Tree("assign", [Tree("access", [tns, *idxs]), expr_node])]):
            arg = _parse_einsum_expr(expr_node)  # type: ignore[arg-type]
            return ein.Einsum(overwrite, ein.Alias(tns.value), tuple(ein.Index(idx.value) for idx in idxs), arg)  # type: ignore[union-attr]

        case _:
            raise ValueError(
                f"Expected top-level assignment or increment, got {tree.data}"
            )
