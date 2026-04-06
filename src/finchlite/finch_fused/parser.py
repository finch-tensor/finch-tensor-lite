"""AI modified: 2026-03-16T14:29:19Z parent=18fbb013175241f5081102d7b13f81f0e6c3de04"""
# AI modified: 2026-03-16T14:40:22Z parent=18fbb013175241f5081102d7b13f81f0e6c3de04

from __future__ import annotations

import ast
import builtins
import inspect
import operator
import textwrap
import types
from collections.abc import Callable
from typing import Any

from . import nodes as fzd

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.MatMult: operator.matmul,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda x, y: x in y,
    ast.NotIn: lambda x, y: x not in y,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}

_BOOL_OPS = {
    ast.And: operator.and_,
    ast.Or: operator.or_,
}

_REV_BIN_OPS = {fn: op for op, fn in _BIN_OPS.items()}
_REV_CMP_OPS = {fn: op for op, fn in _CMP_OPS.items()}
_REV_UNARY_OPS = {fn: op for op, fn in _UNARY_OPS.items()}
_REV_BOOL_OPS = {fn: op for op, fn in _BOOL_OPS.items()}


class _FusedFunctionParser:
    def __init__(self, fn: Callable[..., Any], fn_def: ast.FunctionDef):
        self.fn = fn
        self.fn_def = fn_def
        self.globals = getattr(fn, "__globals__", {})
        self.closurevars: dict[str, Any] = {}
        if hasattr(fn, "__code__") and hasattr(fn, "__closure__") and fn.__closure__:
            for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                self.closurevars[name] = cell.cell_contents
        self.locals: set[str] = set()

    def parse(self) -> fzd.Function:
        if len(self.fn_def.decorator_list) > 1:  # The @jit decorator is in this list
            raise self._unsupported(
                self.fn_def,
                "Decorated functions are not supported in finch_fused parser draft.",
            )

        params = tuple(self._parse_parameter(arg) for arg in self.fn_def.args.args)
        body = self._parse_block(self.fn_def.body)
        return fzd.Function(fzd.Literal(self.fn_def.name), params, body)

    def _parse_parameter(self, arg: ast.arg) -> fzd.Variable:
        if arg.annotation is not None:
            raise self._unsupported(arg, "Annotated parameters are not supported.")
        self.locals.add(arg.arg)
        return fzd.Variable(arg.arg)

    def _parse_block(self, stmts: list[ast.stmt]) -> fzd.Block:
        return fzd.Block(tuple(self._parse_stmt(stmt) for stmt in stmts))

    def _parse_stmt(self, stmt: ast.stmt) -> fzd.FusedStatement:
        match stmt:
            case ast.Assign(targets=[ast.Name(id=name)], value=value):
                rhs = self._parse_expr(value)
                self.locals.add(name)
                return fzd.Assign(fzd.Variable(name), rhs)
            case ast.AugAssign(target=ast.Name(id=name), op=op, value=value):
                lhs = fzd.Variable(name)
                rhs = fzd.BinaryOp(lhs, self._parse_op(op), self._parse_expr(value))
                self.locals.add(name)
                return fzd.Assign(lhs, rhs)
            case ast.Expr(value=value):
                return fzd.ExprStmt(self._parse_expr(value))
            case ast.If(test=test, body=body, orelse=orelse):
                else_block = self._parse_block(orelse) if orelse else None
                return fzd.If(
                    self._parse_expr(test), self._parse_block(body), else_block
                )
            case ast.While(test=test, body=body, orelse=[]):
                return fzd.While(self._parse_expr(test), self._parse_block(body))
            case ast.For(target=ast.Name(id=name), iter=iterable, body=body, orelse=[]):
                self.locals.add(name)
                return fzd.For(
                    fzd.Variable(name),
                    self._parse_expr(iterable),
                    self._parse_block(body),
                )
            case ast.Return(value=None):
                return fzd.Return(())
            case ast.Return(value=value):
                assert value is not None
                match value:
                    case ast.Tuple(elts=elts):
                        return fzd.Return(tuple(self._parse_expr(elt) for elt in elts))
                    case _:
                        return fzd.Return((self._parse_expr(value),))
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                raise self._unsupported(
                    stmt,
                    "Local functions are not supported in finch_fused parser draft.",
                )
            case ast.For(orelse=orelse) if orelse:
                raise self._unsupported(stmt, "For-else blocks are not supported.")
            case ast.While(orelse=orelse) if orelse:
                raise self._unsupported(stmt, "While-else blocks are not supported.")
            case _:
                raise self._unsupported(
                    stmt,
                    f"Unsupported statement type: {type(stmt).__name__}",
                )

    def _parse_expr(self, expr: ast.expr) -> fzd.FusedExpression:
        match expr:
            case ast.Constant(value=value):
                return fzd.Literal(value)
            case ast.Name(id=name):
                return self._parse_name(name)
            case ast.Call(func=func, args=args, keywords=[]):
                return fzd.Call(
                    self._parse_expr(func),
                    tuple(self._parse_expr(arg) for arg in args),
                )
            case ast.BinOp(left=left, op=op, right=right):
                return fzd.BinaryOp(
                    self._parse_expr(left),
                    self._parse_op(op),
                    self._parse_expr(right),
                )
            case ast.Compare(left=left, ops=[op], comparators=[right]):
                return fzd.Compare(
                    self._parse_expr(left),
                    self._parse_op(op),
                    self._parse_expr(right),
                )
            case ast.Compare() as cmp if len(cmp.ops) > 1:
                return self._parse_chained_compare(cmp)
            case ast.BoolOp(op=op, values=values) if len(values) >= 2:
                return self._parse_bool_op(op, values)
            case ast.UnaryOp(op=op, operand=operand):
                return fzd.Call(self._parse_op(op), (self._parse_expr(operand),))
            case ast.Tuple(elts=elts):
                return fzd.Call(
                    fzd.Literal(tuple),
                    tuple(self._parse_expr(elt) for elt in elts),
                )
            case ast.IfExp(test=test, body=body, orelse=orelse):
                return fzd.Call(
                    fzd.Literal(lambda cond, t, f: t if cond else f),
                    (
                        self._parse_expr(test),
                        self._parse_expr(body),
                        self._parse_expr(orelse),
                    ),
                )
            case ast.Attribute(value=value, attr=attr):
                base = self._parse_expr(value)
                return fzd.Call(fzd.Literal(getattr), (base, fzd.Literal(attr)))
            case ast.Break():
                return fzd.Break()  # type: ignore[return-value]
            case _:
                raise self._unsupported(
                    expr,
                    f"Unsupported expression type: {type(expr).__name__}",
                )

    def _parse_name(self, name: str) -> fzd.FusedExpression:
        if name in self.locals:
            return fzd.Variable(name)
        if name in self.closurevars:
            return fzd.Literal(self.closurevars[name])
        if name in self.globals:
            return fzd.Literal(self.globals[name])
        if hasattr(builtins, name):
            return fzd.Literal(getattr(builtins, name))
        return fzd.Variable(name)

    def _parse_op(
        self, op: ast.operator | ast.unaryop | ast.boolop | ast.cmpop
    ) -> fzd.Literal:
        op_type: Any = type(op)
        for table in (_BIN_OPS, _UNARY_OPS, _BOOL_OPS, _CMP_OPS):
            fn = table.get(op_type)
            if fn is not None:
                return fzd.Literal(fn)
        raise self._unsupported(op, f"Unsupported operator type: {type(op).__name__}")

    def _parse_chained_compare(self, cmp: ast.Compare) -> fzd.FusedExpression:
        first = fzd.Compare(
            self._parse_expr(cmp.left),
            self._parse_op(cmp.ops[0]),
            self._parse_expr(cmp.comparators[0]),
        )
        expr: fzd.FusedExpression = first
        for i in range(1, len(cmp.ops)):
            next_cmp = fzd.Compare(
                self._parse_expr(cmp.comparators[i - 1]),
                self._parse_op(cmp.ops[i]),
                self._parse_expr(cmp.comparators[i]),
            )
            expr = fzd.BinaryOp(expr, fzd.Literal(operator.and_), next_cmp)
        return expr

    def _parse_bool_op(
        self, op: ast.boolop, values: list[ast.expr]
    ) -> fzd.FusedExpression:
        expr = self._parse_expr(values[0])
        op_lit = self._parse_op(op)
        for value in values[1:]:
            expr = fzd.BinaryOp(expr, op_lit, self._parse_expr(value))
        return expr

    def _unsupported(self, node: ast.AST, message: str) -> ValueError:
        lineno = getattr(node, "lineno", "?")
        return ValueError(f"{message} (line {lineno})")


def parse_fused_function(fn: Callable[..., Any]) -> fzd.Function:
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    fn_name = getattr(fn, "__name__", None)

    for node in tree.body:
        match node:
            case ast.FunctionDef(name=name) if fn_name is None or name == fn_name:
                return _FusedFunctionParser(fn, node).parse()
            case ast.AsyncFunctionDef(name=name) if fn_name is None or name == fn_name:
                raise ValueError(
                    "Async functions are not supported in finch_fused parser draft."
                )

    raise ValueError("Expected a top-level Python function definition.")


def parse_fused_module(*functions: Callable[..., Any]) -> fzd.Module:
    return fzd.Module(tuple(parse_fused_function(fn) for fn in functions))


def parse_function(fn: Callable[..., Any]) -> fzd.Function:
    return parse_fused_function(fn)


class _FusedToPythonAST:
    def __init__(self) -> None:
        self._bool_fns = tuple(_REV_BOOL_OPS)
        self._extra_globals: dict[str, Any] = {}

    def parse_function(self, function: fzd.Function) -> ast.FunctionDef:
        if not isinstance(function.name.val, str):
            raise ValueError("Fused function name must be a string literal.")

        args = ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg=param.name, annotation=None) for param in function.params
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        )

        body = [self._stmt_to_ast(stmt) for stmt in function.body.body]
        if len(body) == 0:
            body = [ast.Pass()]

        return ast.FunctionDef(  # type: ignore # noqa: PGH003
            name=function.name.val,
            args=args,
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
            type_params=[],
        )

    def parse_module(self, module: fzd.Module) -> ast.Module:
        return ast.Module(
            body=[self.parse_function(function) for function in module.functions],
            type_ignores=[],
        )

    def _stmt_to_ast(self, stmt: fzd.FusedStatement) -> ast.stmt:
        match stmt:
            case fzd.Assign(lhs=fzd.Variable(name=name), rhs=rhs):
                return ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=self._expr_to_ast(rhs),
                )
            case fzd.ExprStmt(value=value):
                return ast.Expr(value=self._expr_to_ast(value))
            case fzd.If(cond=cond, then_body=then_body, else_body=else_body):
                return ast.If(
                    test=self._expr_to_ast(cond),
                    body=self._block_to_ast(then_body),
                    orelse=self._block_to_ast(else_body)
                    if else_body is not None
                    else [],
                )
            case fzd.While(cond=cond, body=body):
                return ast.While(
                    test=self._expr_to_ast(cond),
                    body=self._block_to_ast(body),
                    orelse=[],
                )
            case fzd.For(target=fzd.Variable(name=name), iterable=iterable, body=body):
                return ast.For(
                    target=ast.Name(id=name, ctx=ast.Store()),
                    iter=self._expr_to_ast(iterable),
                    body=self._block_to_ast(body),
                    orelse=[],
                    type_comment=None,
                )
            case fzd.Return(values=values):
                if len(values) == 0:
                    return ast.Return(value=None)
                if len(values) == 1:
                    return ast.Return(value=self._expr_to_ast(values[0]))
                return ast.Return(
                    value=ast.Tuple(
                        elts=[self._expr_to_ast(value) for value in values],
                        ctx=ast.Load(),
                    )
                )
            case _:
                raise ValueError(
                    f"Unsupported fused statement type: {type(stmt).__name__}"
                )

    def _block_to_ast(self, block: fzd.Block) -> list[ast.stmt]:
        body = [self._stmt_to_ast(stmt) for stmt in block.body]
        return body if len(body) > 0 else [ast.Pass()]

    def _expr_to_ast(self, expr: fzd.FusedExpression) -> ast.expr:
        match expr:
            case fzd.Variable(name=name):
                return ast.Name(id=name, ctx=ast.Load())
            case fzd.Literal(val=value):
                return self._literal_to_expr(value)
            case fzd.Call(fn=fn, args=args):
                if isinstance(fn, fzd.Literal) and fn.val is tuple:
                    return ast.Tuple(
                        elts=[self._expr_to_ast(arg) for arg in args],
                        ctx=ast.Load(),
                    )
                return ast.Call(
                    func=self._expr_to_ast(fn),
                    args=[self._expr_to_ast(arg) for arg in args],
                    keywords=[],
                )
            case fzd.Compare(left=left, op=fzd.Literal(val=op_fn), right=right):
                cmp_op = _REV_CMP_OPS.get(op_fn)
                if cmp_op is None:
                    raise ValueError(f"Unsupported fused compare operator: {op_fn!r}")
                return ast.Compare(
                    left=self._expr_to_ast(left),
                    ops=[cmp_op()],
                    comparators=[self._expr_to_ast(right)],
                )
            case fzd.BinaryOp(left=left, op=fzd.Literal(val=op_fn), right=right):
                bool_op = _REV_BOOL_OPS.get(op_fn)
                if bool_op is not None:
                    return ast.BoolOp(
                        op=bool_op(),
                        values=[self._expr_to_ast(left), self._expr_to_ast(right)],
                    )
                bin_op = _REV_BIN_OPS.get(op_fn)
                if bin_op is None:
                    raise ValueError(f"Unsupported fused binary operator: {op_fn!r}")
                return ast.BinOp(
                    left=self._expr_to_ast(left),
                    op=bin_op(),
                    right=self._expr_to_ast(right),
                )
            case _:
                raise ValueError(
                    f"Unsupported fused expression type: {type(expr).__name__}"
                )

    def _literal_to_expr(self, value: Any) -> ast.expr:
        if isinstance(value, (str, bytes, int, float, complex, bool, type(None))):
            return ast.Constant(value=value)

        if callable(value):
            if getattr(builtins, getattr(value, "__name__", ""), None) is value:
                return ast.Name(id=value.__name__, ctx=ast.Load())

            name = getattr(value, "__name__", None)
            if name is not None and name.isidentifier():
                self._extra_globals[name] = value
                return ast.Name(id=name, ctx=ast.Load())

        if isinstance(value, types.ModuleType):
            name = value.__name__
            self._extra_globals[name] = value
            return ast.Name(id=name, ctx=ast.Load())

        raise ValueError(f"Literal cannot be represented in Python AST: {value!r}")


def fused_function_to_python_ast(function: fzd.Function) -> ast.FunctionDef:
    node = _FusedToPythonAST().parse_function(function)
    return ast.fix_missing_locations(node)


def fused_function_to_python_function(function: fzd.Function) -> Callable:
    converter = _FusedToPythonAST()
    func_def = ast.fix_missing_locations(converter.parse_function(function))
    module_node = ast.Module(body=[func_def], type_ignores=[])
    code = compile(module_node, "<fused>", "exec")
    globals_dict: dict[str, Any] = {
        "__builtins__": builtins,
        "operator": operator,
        **converter._extra_globals,
    }
    exec(code, globals_dict)  # noqa: S102
    return globals_dict[function.name.val]


def fused_module_to_python_ast(module: fzd.Module) -> ast.Module:
    node = _FusedToPythonAST().parse_module(module)
    return ast.fix_missing_locations(node)
