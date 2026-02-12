"""
Parser for Finch Assembly in Python's multiline strings.

Supports constructing assembly nodes objects from parsed strings.

There is also a dedicated VS Code extension for proper highlighting
of tagged strings: https://github.com/finch-tensor/vscode-finch-assembly.
The extension is not yet available on VS Code marketplace. You can find
installation file here: https://github.com/finch-tensor/vscode-finch-assembly/releases
"""

import operator

import numpy as np

from lark import Lark, Token, Tree

from . import nodes as asm

assembly_parser = Lark(
    """
    %import common.CNAME
    %import common.INT
    %import common.DECIMAL
    %import common.CPP_COMMENT
    %import common.C_COMMENT
    %import common.NEWLINE
    %import common.WS_INLINE
    %ignore WS_INLINE

    _FINCH: "finch" | "finch-asm"
    _NEWLINE: NEWLINE
    _COMMENT: C_COMMENT | CPP_COMMENT
    OP: "+" | "-" | "*" | "or" | "and" | "|" | "&" | "^" | "<<" | ">>"
      | "//" | "/" | "%" | "**" | ">" | "<" | ">=" | "<=" | "==" | "!="

    start: _FINCH _NEWLINE+ block
    block: (_stmt _NEWLINE+)* _stmt
    _stmt: assign
         | increment
         | for_loop
         | if
         | if_else
         | resize
         | _COMMENT
    ?access_expr: access_expr OP access_expr | CNAME | INT
    access: CNAME "[" access_expr "]"
    ?expr: CNAME | INT | DECIMAL | access | expr OP expr
    ?lhs: CNAME | access
    assign: lhs "=" expr
    increment: lhs OP "=" expr
    resize: "resize" "(" CNAME "," access_expr ")"
    for_loop: "for" "(" CNAME "in" access_expr ":" access_expr ")" _NEWLINE+ block _NEWLINE+ "end"
    if: "if" "(" expr ")" _NEWLINE+ block _NEWLINE+ "end"
    if_else: "if" "(" expr ")" _NEWLINE+ block _NEWLINE+ "else" _NEWLINE+ block _NEWLINE+ "end"
"""  # noqa: E501
)

_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def parse_assembly(
    code: str, vars: dict[str, asm.Variable], position_type: type = np.intp
) -> asm.AssemblyStatement:
    """
    Parse Finch Assembly code and convert it to assembly node objects.

    Takes a string containing Finch Assembly code and transforms it into a structured
    representation using assembly nodes. The parser supports assignments, increments,
    for loops, if/if-else statements, array accesses, and arithmetic/logical operations.

    Args:
        code: The Finch Assembly code to parse. Should start with "finch" or "finch-asm"
            followed by assembly statements. Comments (C/C++ style) are supported.
        vars: Dictionary mapping variable names (as strings) to Variable objects.
            Used to resolve variable references in the assembly code.
        position_type: NumPy integer type to use for integer literals. Affects the dtype
            of parsed integer constants. (default: np.intp)

    Returns:
        A Finch Assembly Block representing the parsed code.

    Raises:
        Exception: If the parser encounters unrecognized syntax or tree nodes.

    Example:
        >>> from finchlite.finch_assembly import nodes as asm
        >>> import numpy as np
        >>> vars = {"i": asm.Variable("i", int), "arr": asm.Variable("arr", np.ndarray)}
        >>> code = '''finch
        ... arr[i] = 42
        ... '''
        >>> stmt = parse_assembly(code, vars)
    """
    tree = assembly_parser.parse(code.strip())

    def ctx(tree: Tree):
        match tree:
            case Token("CNAME", val):
                return vars[val]
            case Token("OP", val):
                return _OPS[val]
            case Token("INT", val):
                return asm.Literal(position_type(val))
            case Token("DECIMAL", val):
                return asm.Literal(float(val))
            case Tree("start", [Tree("block", bodies)]):
                return asm.Block(tuple(ctx(b) for b in bodies))
            case Tree("for_loop", [i, start, stop, Tree("block", bodies)]):
                return asm.ForLoop(
                    ctx(i),
                    ctx(start),
                    ctx(stop),
                    asm.Block(tuple(ctx(b) for b in bodies)),
                )
            case Tree("if", [cond, Tree("block", bodies)]):
                return asm.If(ctx(cond), asm.Block(tuple(ctx(b) for b in bodies)))
            case Tree(
                "if_else", [cond, Tree("block", bodies), Tree("block", else_bodies)]
            ):
                return asm.IfElse(
                    ctx(cond),
                    asm.Block(tuple(ctx(b) for b in bodies)),
                    asm.Block(tuple(ctx(b) for b in else_bodies)),
                )
            case Tree("resize", [arr, size]):
                return asm.Call(asm.Literal(np.resize), (ctx(arr), ctx(size)))
            case Tree("assign", [lhs, expr]):
                return asm.Assign(ctx(lhs), ctx(expr))
            case Tree("access", [tns, access_expr]):
                return asm.Load(ctx(tns), ctx(access_expr))
            case Tree("increment", [Tree("access", [tns, access_expr]), op, expr]):
                expr_e: asm.AssemblyExpression = ctx(access_expr)
                tns_e: asm.Slot = ctx(tns)
                return asm.Store(
                    tns_e,
                    expr_e,
                    asm.Call(
                        asm.Literal(ctx(op)), (asm.Load(tns_e, expr_e), ctx(expr))
                    ),
                )
            case Tree("expr" | "access_expr", [expr1, op, expr2]):
                return asm.Call(asm.Literal(ctx(op)), (ctx(expr1), ctx(expr2)))
            case other:
                raise Exception(f"{other} not recognized.")

    return ctx(tree)
