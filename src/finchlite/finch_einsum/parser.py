from typing import Any

from lark import Lark, Tree

from ..algebra import ffuncs
from ..symbolic import Namespace
from . import nodes as ein

nary_ops = {
    "+": ffuncs.add,
    "add": ffuncs.add,
    "-": ffuncs.sub,
    "sub": ffuncs.sub,
    "subtract": ffuncs.sub,
    "*": ffuncs.mul,
    "mul": ffuncs.mul,
    "multiply": ffuncs.mul,
    "/": ffuncs.truediv,
    "div": ffuncs.truediv,
    "divide": ffuncs.truediv,
    "//": ffuncs.floordiv,
    "fld": ffuncs.floordiv,
    "floor_divide": ffuncs.floordiv,
    "%": ffuncs.mod,
    "mod": ffuncs.mod,
    "remainder": ffuncs.mod,
    "**": ffuncs.pow,
    "pow": ffuncs.pow,
    "power": ffuncs.pow,
    "==": ffuncs.eq,
    "eq": ffuncs.eq,
    "equal": ffuncs.eq,
    "!=": ffuncs.ne,
    "ne": ffuncs.ne,
    "not_equal": ffuncs.ne,
    "<": ffuncs.lt,
    "lt": ffuncs.lt,
    "less": ffuncs.lt,
    "<=": ffuncs.le,
    "le": ffuncs.le,
    "less_equal": ffuncs.le,
    ">": ffuncs.gt,
    "gt": ffuncs.gt,
    "greater": ffuncs.gt,
    ">=": ffuncs.ge,
    "ge": ffuncs.ge,
    "greater_equal": ffuncs.ge,
    "&": ffuncs.and_,
    "bitwise_and": ffuncs.and_,
    "|": ffuncs.or_,
    "bitwise_or": ffuncs.or_,
    "^": ffuncs.xor,
    "bitwise_xor": ffuncs.xor,
    "<<": ffuncs.lshift,
    "lshift": ffuncs.lshift,
    "bitwise_left_shift": ffuncs.lshift,
    ">>": ffuncs.rshift,
    "rshift": ffuncs.rshift,
    "bitwise_right_shift": ffuncs.rshift,
    "and": ffuncs.logical_and,
    "or": ffuncs.logical_or,
    "not": ffuncs.logical_not,
    "min": ffuncs.min,
    "max": ffuncs.max,
    "logaddexp": ffuncs.logaddexp,
}


unary_ops = {
    "+": ffuncs.pos,
    "pos": ffuncs.pos,
    "positive": ffuncs.pos,
    "-": ffuncs.neg,
    "neg": ffuncs.neg,
    "negative": ffuncs.neg,
    "~": ffuncs.invert,
    "invert": ffuncs.invert,
    "bitwise_invert": ffuncs.invert,
    "not": ffuncs.logical_not,
    "logical_not": ffuncs.logical_not,
    "abs": ffuncs.abs,
    "absolute": ffuncs.abs,
    "sqrt": ffuncs.sqrt,
    "exp": ffuncs.exp,
    "log": ffuncs.log,
    "log1p": ffuncs.log1p,
    "log10": ffuncs.log10,
    "log2": ffuncs.log2,
    "sin": ffuncs.sin,
    "cos": ffuncs.cos,
    "tan": ffuncs.tan,
    "sinh": ffuncs.sinh,
    "cosh": ffuncs.cosh,
    "tanh": ffuncs.tanh,
    "asin": ffuncs.arcsin,
    "arcsin": ffuncs.arcsin,
    "acos": ffuncs.arccos,
    "arccos": ffuncs.arccos,
    "atan": ffuncs.arctan,
    "arctan": ffuncs.arctan,
    "asinh": ffuncs.arcsinh,
    "arcsinh": ffuncs.arcsinh,
    "acosh": ffuncs.arccosh,
    "arccosh": ffuncs.arccosh,
    "atanh": ffuncs.arctanh,
    "arctanh": ffuncs.arctanh,
}


reduction_ops = {
    "+": ffuncs.add,
    "add": ffuncs.add,
    "sum": ffuncs.add,
    "*": ffuncs.mul,
    "mul": ffuncs.mul,
    "prod": ffuncs.mul,
    "and": ffuncs.logical_and,
    "all": ffuncs.logical_and,
    "or": ffuncs.logical_or,
    "any": ffuncs.logical_or,
    "min": ffuncs.min,
    "minimum": ffuncs.min,
    "max": ffuncs.max,
    "maximum": ffuncs.max,
    "&": ffuncs.and_,
    "bitwise_and": ffuncs.and_,
    "|": ffuncs.or_,
    "bitwise_or": ffuncs.or_,
    "^": ffuncs.xor,
    "bitwise_xor": ffuncs.xor,
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


def _parse_einop_expr(t: Tree) -> ein.EinsumExpression:
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
        case Tree("access", [tns, *idxs]):
            return ein.Access(
                ein.Alias(tns.value),  # type: ignore[union-attr]
                tuple(ein.Index(idx.value) for idx in idxs),  # type: ignore[union-attr]
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


def parse_einop(expr: str) -> ein.EinsumNode:
    tree = lark_parser.parse(expr)
    match tree:
        case Tree(
            "start",
            [Tree("increment", [Tree("access", [tns, *idxs]), op_token, expr_node])],
        ):
            arg = _parse_einop_expr(expr_node)  # type: ignore[arg-type]
            idxs_exprs = tuple(ein.Index(idx.value) for idx in idxs)  # type: ignore[union-attr]
            op = ein.Literal(reduction_ops[op_token.value])  # type: ignore[union-attr]
            return ein.Einsum(
                op,
                ein.Alias(tns.value),  # type: ignore[union-attr]
                idxs_exprs,
                arg,  # type: ignore[union-attr]
            )

        case Tree("start", [Tree("assign", [Tree("access", [tns, *idxs]), expr_node])]):
            arg = _parse_einop_expr(expr_node)  # type: ignore[arg-type]
            op = ein.Literal(ffuncs.overwrite)
            return ein.Einsum(
                op,
                ein.Alias(tns.value),  # type: ignore[union-attr]
                tuple(ein.Index(idx.value) for idx in idxs),  # type: ignore[union-attr]
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
        op = ein.Literal(ffuncs.overwrite)
    else:
        op = ein.Literal(ffuncs.add)
    out_tns = ein.Alias(spc.freshen("B"))
    idxs = tuple(ein.Index(j) for j in output_idxs)
    in_tnss = [ein.Alias(spc.freshen("A")) for _ in operands]
    arg = ein.Access(in_tnss[0], tuple(ein.Index(i) for i in input_idxs[0]))
    for i in range(1, len(operands)):
        arg = ein.Call(
            ein.Literal(ffuncs.mul),
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
