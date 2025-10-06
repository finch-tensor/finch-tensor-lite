import operator
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self, cast

import numpy as np

import finchlite.finch_logic as lgc
from finchlite.algebra import (
    init_value,
    is_commutative,
    overwrite,
    promote_max,
    promote_min,
)
from finchlite.finch_logic import LogicNode
from finchlite.symbolic import Term, TermTree


@dataclass(eq=True, frozen=True)
class EinsumExpr(Term, ABC):
    """
    EinsumExpr

    Represents a pointwise expression in the Einsum IR
    """

    @classmethod
    def head(cls):
        """Returns the head of the node."""
        return cls

    @classmethod
    def make_term(cls, head, *children: Term) -> Self:
        return head.from_children(*children)

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        return cls(*children)

    def __str__(self):
        ctx = EinsumPrinterContext()
        return ctx.print_pointwise(self)


@dataclass(eq=True, frozen=True)
class Index(EinsumExpr):
    """
    Represents a  AST expression for an index named `name`.

    Attributes:
        name: The name of the index.
    """

    name: str


@dataclass(eq=True, frozen=True)
class Alias(EinsumExpr):
    """
    Represents a  AST expression for an index named `name`.

    Attributes:
        name: The name of the index.
    """

    name: str


@dataclass(eq=True, frozen=True)
class Access(EinsumExpr, TermTree):
    """
    Access

    Tensor access like a[i, j].

    Attributes:
        tensor: The tensor to access.
        idxs: The indices at which to access the tensor.
    """

    tns: EinsumExpr
    idxs: tuple[EinsumExpr, ...]  # (Field('i'), Field('j'))
    # Children: None (leaf)

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # First child is tns, rest are indices
        if len(children) < 1:
            raise ValueError("Access expects at least 1 child")
        tns = cast(EinsumExpr, children[0])
        idxs = cast(tuple[EinsumExpr, ...], children[1:])
        return cls(tns, tuple(idxs))

    @property
    def children(self):
        return [self.tns, *self.idxs]


@dataclass(eq=True, frozen=True)
class Call(EinsumExpr, TermTree):
    """
    Call

    Represents an operation like + or * on pointwise expressions for multiple operands.
    If operation is not commutative, pointwise node must be binary, with 2 args at most.

    Attributes:
        op: The function to apply e.g.,
            operator.add, operator.mul, operator.subtract, operator.div, etc...
            Must be a callable.
        args: The arguments to the operation.
    """

    op: Callable  # the function to apply e.g., operator.add
    args: tuple[EinsumExpr, ...]  # Subtrees
    # input_fields: tuple[tuple[Field, ...], ...]
    # Children: The args

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # First child is op, rest are args
        if len(children) < 2:
            raise ValueError("Call expects at least 2 children (op + 1 arg)")
        op = cast(Callable, children[0])
        args = cast(tuple[EinsumExpr, ...], children[1:])
        return cls(op, tuple(args))

    @property
    def children(self):
        return [self.op, *self.args]


@dataclass(eq=True, frozen=True)
class Literal(EinsumExpr):
    """
    Literal

    A scalar literal/value for pointwise operations.
    """

    val: float

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        return isinstance(other, Literal) and self.val == other.val


@dataclass(eq=True, frozen=True)
class Einsum(EinsumExpr, TermTree):
    """
    Einsum

    A einsum operation that maps pointwise expressions and aggregates them.

    Attributes:
        op: The function to apply to the pointwise expressions
                    (e.g. +=, f=, max=, etc...). Must be a callable.

        idxs: The indices that are used in the output
                    (i.e. i, j).

        arg: The pointwise expression that
                    is mapped and aggregated.
    """

    # technically a reduce operation, much akin to the one in aggregate
    op: Callable

    output: EinsumExpr
    idxs: tuple[EinsumExpr, ...]
    arg: EinsumExpr

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # Expecting exactly 4 children
        if len(children) != 4:
            raise ValueError(f"Einsum expects 4 children, got {len(children)}")
        op = cast(Callable, children[0])
        output = cast(EinsumExpr, children[1])
        idxs = cast(tuple[EinsumExpr, ...], children[2])
        arg = cast(EinsumExpr, children[3])
        return cls(op, output, idxs, arg)

    @property
    def children(self):
        return [self.op, self.output, self.idxs, self.arg]

    def rename(self, new_alias: str):
        return Einsum(
            self.op,
            Alias(new_alias),
            self.idxs,
            self.arg,
        )

    def reorder(self, idxs: tuple[lgc.Field, ...]):
        return Einsum(
            self.op,
            self.output,
            tuple(Index(idx.name) for idx in idxs),
            self.arg,
        )


@dataclass(eq=True, frozen=True)
class Plan(EinsumExpr):
    """
    Plan

    A plan that contains einsum operations.
    Basically a list of einsums and some return values.
    """

    bodies: tuple[Einsum, ...] = ()
    returnValues: tuple[EinsumExpr, ...] = ()

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # The last child is the returnValues tuple, all others are bodies
        if len(children) < 1:
            raise ValueError("Plan expects at least 1 child")
        *bodies, returnValues = children

        return cls(
            tuple(cast(Einsum, b) for b in bodies),
            cast(tuple[EinsumExpr, ...], returnValues),
        )

    @property
    def children(self):
        return [*self.bodies, self.returnValues]

    def __str__(self):
        ctx = EinsumPrinterContext()
        return ctx(self)


class LogicLowerer:
    alias_counter: int = 0

    def __call__(self, prgm: lgc.Plan) -> tuple[Plan, dict[str, lgc.Table]]:
        parameters: dict[str, lgc.Table] = {}
        definitions: dict[str, Einsum] = {}
        return self.compile_plan(prgm, parameters, definitions), parameters

    def get_next_alias(self) -> str:
        self.alias_counter += 1
        return f"einsum_{self.alias_counter}"

    def rename_einsum(
        self, einsum: Einsum, new_alias: str, definitions: dict[str, Einsum]
    ) -> Einsum:
        definitions[new_alias] = einsum
        return einsum.rename(new_alias)

    def compile_plan(
        self,
        plan: lgc.Plan,
        parameters: dict[str, lgc.Table],
        definitions: dict[str, Einsum],
    ) -> Plan:
        einsums: list[Einsum] = []
        returnValue: list[EinsumExpr] = []

        for body in plan.bodies:
            match body:
                case lgc.Plan(_):
                    inner_plan = self.compile_plan(body, parameters, definitions)
                    einsums.extend(inner_plan.bodies)
                    break
                case lgc.Query(lgc.Alias(name), lgc.Table(_, _)):
                    parameters[name] = body.rhs
                case lgc.Query(lgc.Alias(name), rhs):
                    einsums.append(
                        self.rename_einsum(
                            self.lower_to_einsum(rhs, einsums, parameters, definitions),
                            name,
                            definitions,
                        )
                    )
                case lgc.Produces(args):
                    returnValue = [
                        Alias(arg.name)
                        if isinstance(arg, lgc.Alias)
                        else self.lower_to_einsum(arg, einsums, parameters, definitions)
                        for arg in args
                    ]
                    break
                case _:
                    einsums.append(
                        self.rename_einsum(
                            self.lower_to_einsum(
                                body, einsums, parameters, definitions
                            ),
                            self.get_next_alias(),
                            definitions,
                        )
                    )

        return Plan(tuple(einsums), tuple(returnValue))

    def lower_to_einsum(
        self,
        ex: LogicNode,
        einsums: list[Einsum],
        parameters: dict[str, lgc.Table],
        definitions: dict[str, Einsum],
    ) -> Einsum:
        match ex:
            case lgc.Plan(_):
                raise Exception("Plans within plans are not supported.")
            case lgc.MapJoin(lgc.Literal(operation), args):
                args_list = [
                    self.lower_to_pointwise(arg, einsums, parameters, definitions)
                    for arg in args
                ]
                arg = self.lower_to_pointwise_op(operation, tuple(args_list))
                return Einsum(
                    op=overwrite,
                    output=Alias(self.get_next_alias()),
                    idxs=tuple(Index(field.name) for field in ex.fields),
                    arg=arg,
                )
            case lgc.Reorder(arg, idxs):
                return self.lower_to_einsum(
                    arg, einsums, parameters, definitions
                ).reorder(idxs)
            case lgc.Aggregate(lgc.Literal(operation), lgc.Literal(init), arg, idxs):
                if init != init_value(operation, type(init)):
                    raise Exception(f"""
                    Init value {init} is not the default value
                    for operation {operation} of type {type(init)}.
                    Non standard init values are not supported.
                    """)
                aggregate_expr = self.lower_to_pointwise(
                    arg, einsums, parameters, definitions
                )
                return Einsum(
                    operation,
                    Alias(self.get_next_alias()),
                    tuple(Index(field.name) for field in ex.fields),
                    aggregate_expr,
                )
            case _:
                raise Exception(f"Unrecognized logic: {ex}")

    def lower_to_pointwise_op(
        self, operation: Callable, args: tuple[EinsumExpr, ...]
    ) -> Call:
        # if operation is commutative, we simply pass
        # all the args to the pointwise op since
        # order of args does not matter
        if is_commutative(operation):

            def flatten_args(
                m_args: tuple[EinsumExpr, ...],
            ) -> tuple[EinsumExpr, ...]:
                ret_args: list[EinsumExpr] = []
                for arg in m_args:
                    match arg:
                        case Call(op2, _) if op2 == operation:
                            ret_args.extend(flatten_args(arg.args))
                        case _:
                            ret_args.append(arg)
                return tuple(ret_args)

            return Call(operation, flatten_args(args))

        # combine args from left to right (i.e a / b / c -> (a / b) / c)
        assert len(args) > 1
        result = Call(operation, (args[0], args[1]))
        for arg in args[2:]:
            result = Call(operation, (result, arg))
        return result

    # lowers nested mapjoin logic IR nodes into a single pointwise expression
    def lower_to_pointwise(
        self,
        ex: lgc.LogicNode,
        einsums: list[Einsum],
        parameters: dict[str, lgc.Table],
        definitions: dict[str, Einsum],
    ) -> EinsumExpr:
        match ex:
            case lgc.Reorder(arg, idxs):
                return self.lower_to_pointwise(arg, einsums, parameters, definitions)
            case lgc.MapJoin(lgc.Literal(operation), args):
                args_list = [
                    self.lower_to_pointwise(arg, einsums, parameters, definitions)
                    for arg in args
                ]
                return self.lower_to_pointwise_op(operation, tuple(args_list))
            case lgc.Relabel(
                lgc.Alias(name), idxs
            ):  # relable is really just a glorified pointwise access
                return Access(
                    tns=Alias(name),
                    idxs=tuple(Index(idx.name) for idx in idxs),
                )
            case lgc.Literal(value):
                return Literal(val=value)
            case lgc.Aggregate(
                _, _, _, _
            ):  # aggregate has to be computed seperatley as it's own einsum
                aggregate_einsum_alias = self.get_next_alias()
                einsums.append(
                    self.rename_einsum(
                        self.lower_to_einsum(ex, einsums, parameters, definitions),
                        aggregate_einsum_alias,
                        definitions,
                    )
                )
                return Access(
                    tns=Alias(aggregate_einsum_alias),
                    idxs=tuple(Index(field.name) for field in ex.fields),
                )
            case _:
                raise Exception(f"Unrecognized logic: {ex}")


class EinsumPrinterContext:
    def print_reducer(self, reducer: Callable):
        str_map = {
            overwrite: "=",
            operator.add: "+=",
            operator.sub: "-=",
            operator.mul: "*=",
            operator.truediv: "/=",
            operator.mod: "%=",
            operator.pow: "**=",
            operator.and_: "&=",
            operator.or_: "|=",
            operator.xor: "^=",
            operator.floordiv: "//=",
            operator.mod: "%=",
            operator.pow: "**=",
            promote_max: "max=",
            promote_min: "min=",
        }
        return str_map[reducer]

    def print_pointwise_op_callable(self, op: Callable):
        str_map = {
            operator.add: "+",
            operator.sub: "-",
            operator.mul: "*",
            operator.truediv: "/",
            operator.mod: "%",
            operator.pow: "**",
        }
        return str_map[op]

    def print_pointwise_op(self, pointwise_op: Call):
        opstr = f" {self.print_pointwise_op_callable(pointwise_op.op)} "
        if not is_commutative(pointwise_op.op):
            return f"({pointwise_op.args[0]}{opstr}{pointwise_op.args[1]})"

        return f"({opstr.join(self.print_pointwise(arg) for arg in pointwise_op.args)})"

    def print_indicies(self, idxs: tuple[EinsumExpr, ...]):
        return ", ".join([self.print_pointwise(idx) for idx in idxs])

    def print_pointwise(self, arg: EinsumExpr):
        match arg:
            case Einsum(_, _, _, _):
                return self.print_einsum(arg)
            case Alias(name):
                return name
            case Index(name):
                return name
            case Access(tns, idxs):
                return f"{self.print_pointwise(tns)}[{self.print_indicies(idxs)}]"
            case Call(_, __):
                return self.print_pointwise_op(arg)
            case Literal(val):
                return str(val)

    def print_einsum(self, einsum: Einsum) -> str:
        return (
            f"{self.print_pointwise(einsum.output)}["
            f"{self.print_indicies(einsum.idxs)}] "
            f"{self.print_reducer(einsum.op)} "
            f"{self.print_pointwise(einsum.arg)}"
        )

    def print_einsum_plan(self, einsum_plan: Plan) -> str:
        statements = "\n".join(
            [self.print_einsum(statement) for statement in einsum_plan.bodies]
        )
        return_values = ", ".join(
            [
                self.print_pointwise(return_value)
                for return_value in einsum_plan.returnValues
            ]
        )
        return f"{statements}\nreturn {return_values}"

    def __call__(self, prgm: Plan) -> str:
        return self.print_einsum_plan(prgm)


class EinsumInterpreter:
    def __call__(self, einsum_plan: Plan, parameters: dict[str, lgc.Table]):
        return self.print(einsum_plan, parameters)

    def print(self, einsum_plan: Plan, parameters: dict[str, lgc.Table]):
        for str, table in parameters.items():
            print(f"Parameter: {str} = {table}")

        print(einsum_plan)
        return (np.arange(6, dtype=np.float32).reshape(2, 3),)


class EinsumCompiler:
    def __init__(self):
        self.el = LogicLowerer()

    def __call__(self, prgm: lgc.Plan):
        einsum_plan, parameters = self.el(prgm)

        return einsum_plan, parameters


class EinsumScheduler:
    def __init__(self, ctx: EinsumCompiler):
        self.ctx = ctx
        self.interpret = EinsumInterpreter()

    def __call__(self, prgm: LogicNode):
        if not isinstance(prgm, lgc.Plan):
            raise TypeError(f"EinsumScheduler expects a Plan, got {type(prgm)}")
        einsum_plan, parameters = self.ctx(prgm)
        return self.interpret(einsum_plan, parameters)
