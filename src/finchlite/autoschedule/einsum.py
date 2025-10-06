import operator
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self, cast

import numpy as np

from finchlite.algebra import (
    init_value,
    is_commutative,
    overwrite,
    promote_max,
    promote_min,
)
from finchlite.finch_logic import Alias, Field, Literal, LogicNode, Plan, Query, Relabel
from finchlite.finch_logic.nodes import Aggregate, MapJoin, Produces, Reorder, Table
from finchlite.symbolic import Term, TermTree


@dataclass(eq=True, frozen=True)
class PointwiseNode(Term, ABC):
    """
    PointwiseNode

    Represents an AST node in the Einsum Pointwise Expression IR
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
class PointwiseNamedField(PointwiseNode):
    """
    PointwiseNamedFiled

    Could be an alias to a tensor, could be a named field in an index array
    """

    name: str

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # Expecting a single child which is the name
        if len(children) != 1:
            raise ValueError(f"PointwiseNamedField expects 1 child got {len(children)}")
        return cls(str(children[0]))

    @property
    def children(self):
        return [self.name]


@dataclass(eq=True, frozen=True)
class PointwiseAccess(PointwiseNode, TermTree):
    """
    PointwiseAccess

    Tensor access like a[i, j].

    Attributes:
        tensor: The tensor to access.
        idxs: The indices at which to access the tensor.
    """

    alias: PointwiseNode
    idxs: tuple[PointwiseNode, ...]  # (Field('i'), Field('j'))
    # Children: None (leaf)

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # First child is alias, rest are indices
        if len(children) < 1:
            raise ValueError("PointwiseAccess expects at least 1 child")
        alias = cast(PointwiseNode, children[0])
        idxs = cast(tuple[PointwiseNode, ...], children[1:])
        return cls(alias, tuple(idxs))

    @property
    def children(self):
        return [self.alias, *self.idxs]


@dataclass(eq=True, frozen=True)
class GetSparseCoordArray(PointwiseNode, TermTree):
    """
    GetSparseCoordArray

    Gets the coordinate array of a sparse tensor stored in COO form

    Attributes:
        sparse_tensor: The sparse tensor to access
    """

    sparse_tensor: PointwiseNode
    dimension: PointwiseNamedField | None

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        if len(children) != 1 and len(children) != 2:
            raise ValueError(
                f"GetSparseCoordArray expects 1 or 2 children, \
                             got {len(children)}"
            )
        return cls(
            cast(PointwiseNode, children[0]),
            cast(PointwiseNamedField | None, children[1])
            if len(children) == 2
            else None,
        )

    @property
    def children(self):
        return [self.sparse_tensor, self.dimension]


@dataclass(eq=True, frozen=True)
class GetSparseValueArray(PointwiseNode, TermTree):
    """
    GetSparseValueArray

    Gets the value array of a sparse tensor stored in COO form

    Attributes:
        sparse_tensor: The sparse tensor to access
    """

    sparse_tensor: PointwiseNode

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        if len(children) != 1:
            raise ValueError(f"GetSparseValueArray expects 1 child got {len(children)}")
        return cls(cast(PointwiseNode, children[0]))

    @property
    def children(self):
        return [self.sparse_tensor]


@dataclass(eq=True, frozen=True)
class PointwiseOp(PointwiseNode, TermTree):
    """
    PointwiseOp

    Represents an operation like + or * on pointwise expressions for multiple operands.
    If operation is not commutative, pointwise node must be binary, with 2 args at most.

    Attributes:
        op: The function to apply e.g.,
            operator.add, operator.mul, operator.subtract, operator.div, etc...
            Must be a callable.
        args: The arguments to the operation.
    """

    op: Callable  # the function to apply e.g., operator.add
    args: tuple[PointwiseNode, ...]  # Subtrees
    # input_fields: tuple[tuple[Field, ...], ...]
    # Children: The args

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # First child is op, rest are args
        if len(children) < 2:
            raise ValueError("PointwiseOp expects at least 2 children (op + 1 arg)")
        op = cast(Callable, children[0])
        args = cast(tuple[PointwiseNode, ...], children[1:])
        return cls(op, tuple(args))

    @property
    def children(self):
        return [self.op, *self.args]


@dataclass(eq=True, frozen=True)
class PointwiseIfElse(PointwiseNode, TermTree):
    """
    PointwiseIfElse

    Attributes:
        condition: The condition to check.
        then_expr: The expression to evaluate if the condition is not zero.
        else_expr: The expression to evaluate if the condition is zero.
    """

    condition: PointwiseNode
    then_expr: PointwiseNode
    else_expr: PointwiseNode

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        if len(children) != 3:
            raise ValueError(f"PointwiseIfElse expects 3 children, got {len(children)}")
        return cls(
            cast(PointwiseNode, children[0]),
            cast(PointwiseNode, children[1]),
            cast(PointwiseNode, children[2]),
        )

    @property
    def children(self):
        return [self.condition, self.then_expr, self.else_expr]


@dataclass(eq=True, frozen=True)
class PointwiseLiteral(PointwiseNode):
    """
    PointwiseLiteral

    A scalar literal/value for pointwise operations.
    """

    val: float

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        return isinstance(other, PointwiseLiteral) and self.val == other.val


# einsum and einsum ast not part of logic IR
# transform to it's own language
@dataclass(eq=True, frozen=True)
class Einsum(PointwiseNode, TermTree):
    """
    Einsum

    A einsum operation that maps pointwise expressions and aggregates them.

    Attributes:
        updateOp: The function to apply to the pointwise expressions
                    (e.g. +=, f=, max=, etc...). Must be a callable.

        input_fields: The indices that are used in the pointwise
                    expression (i.e. i, j, k).

        output_fields: The indices that are used in the output
                    (i.e. i, j).

        pointwise_expr: The pointwise expression that
                    is mapped and aggregated.
    """

    # technically a reduce operation, much akin to the one in aggregate
    reduceOp: Callable

    output: PointwiseNode
    output_fields: tuple[PointwiseNode, ...]
    pointwise_expr: PointwiseNode

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # Expecting exactly 4 children
        if len(children) != 4:
            raise ValueError(f"Einsum expects 4 children, got {len(children)}")
        reduceOp = cast(Callable, children[0])
        output = cast(PointwiseNode, children[1])
        output_fields = cast(tuple[PointwiseNode, ...], children[2])
        pointwise_expr = cast(PointwiseNode, children[3])
        return cls(reduceOp, output, output_fields, pointwise_expr)

    @property
    def children(self):
        return [self.reduceOp, self.output, self.output_fields, self.pointwise_expr]

    def rename(self, new_alias: str):
        return Einsum(
            self.reduceOp,
            PointwiseNamedField(new_alias),
            self.output_fields,
            self.pointwise_expr,
        )

    def reorder(self, idxs: tuple[Field, ...]):
        return Einsum(
            self.reduceOp,
            self.output,
            tuple(PointwiseNamedField(idx.name) for idx in idxs),
            self.pointwise_expr,
        )


@dataclass(eq=True, frozen=True)
class EinsumPlan(PointwiseNode):
    """
    EinsumPlan

    A plan that contains einsum operations.
    Basically a list of einsums and some return values.
    """

    bodies: tuple[Einsum, ...] = ()
    returnValues: tuple[PointwiseNode, ...] = ()

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # The last child is the returnValues tuple, all others are bodies
        if len(children) < 1:
            raise ValueError("EinsumPlan expects at least 1 child")
        *bodies, returnValues = children

        return cls(
            tuple(cast(Einsum, b) for b in bodies),
            cast(tuple[PointwiseNode, ...], returnValues),
        )

    @property
    def children(self):
        return [*self.bodies, self.returnValues]

    def __str__(self):
        ctx = EinsumPrinterContext()
        return ctx(self)


class EinsumLowerer:
    alias_counter: int = 0

    def __call__(self, prgm: Plan) -> tuple[EinsumPlan, dict[str, Table]]:
        parameters: dict[str, Table] = {}
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
        self, plan: Plan, parameters: dict[str, Table], definitions: dict[str, Einsum]
    ) -> EinsumPlan:
        einsums: list[Einsum] = []
        returnValue: list[PointwiseNode] = []

        for body in plan.bodies:
            match body:
                case Plan(_):
                    inner_plan = self.compile_plan(body, parameters, definitions)
                    einsums.extend(inner_plan.bodies)
                    break
                case Query(Alias(name), Table(_, _)):
                    parameters[name] = body.rhs
                case Query(Alias(name), rhs):
                    einsums.append(
                        self.rename_einsum(
                            self.lower_to_einsum(rhs, einsums, parameters, definitions),
                            name,
                            definitions,
                        )
                    )
                case Produces(args):
                    returnValue = [
                        PointwiseNamedField(arg.name)
                        if isinstance(arg, Alias)
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

        return EinsumPlan(tuple(einsums), tuple(returnValue))

    def lower_to_einsum(
        self,
        ex: LogicNode,
        einsums: list[Einsum],
        parameters: dict[str, Table],
        definitions: dict[str, Einsum],
    ) -> Einsum:
        match ex:
            case Plan(_):
                raise Exception("Plans within plans are not supported.")
            case MapJoin(Literal(operation), args):
                args_list = [
                    self.lower_to_pointwise(arg, einsums, parameters, definitions)
                    for arg in args
                ]
                pointwise_expr = self.lower_to_pointwise_op(operation, tuple(args_list))
                return Einsum(
                    reduceOp=overwrite,
                    output=PointwiseNamedField(self.get_next_alias()),
                    output_fields=tuple(
                        PointwiseNamedField(field.name) for field in ex.fields
                    ),
                    pointwise_expr=pointwise_expr,
                )
            case Reorder(arg, idxs):
                return self.lower_to_einsum(
                    arg, einsums, parameters, definitions
                ).reorder(idxs)
            case Aggregate(Literal(operation), Literal(init), arg, idxs):
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
                    PointwiseNamedField(self.get_next_alias()),
                    tuple(PointwiseNamedField(field.name) for field in ex.fields),
                    aggregate_expr,
                )
            case _:
                raise Exception(f"Unrecognized logic: {ex}")

    def lower_to_pointwise_op(
        self, operation: Callable, args: tuple[PointwiseNode, ...]
    ) -> PointwiseOp:
        # if operation is commutative, we simply pass
        # all the args to the pointwise op since
        # order of args does not matter
        if is_commutative(operation):

            def flatten_args(
                m_args: tuple[PointwiseNode, ...],
            ) -> tuple[PointwiseNode, ...]:
                ret_args: list[PointwiseNode] = []
                for arg in m_args:
                    match arg:
                        case PointwiseOp(op2, _) if op2 == operation:
                            ret_args.extend(flatten_args(arg.args))
                        case _:
                            ret_args.append(arg)
                return tuple(ret_args)

            return PointwiseOp(operation, flatten_args(args))

        # combine args from left to right (i.e a / b / c -> (a / b) / c)
        assert len(args) > 1
        result = PointwiseOp(operation, (args[0], args[1]))
        for arg in args[2:]:
            result = PointwiseOp(operation, (result, arg))
        return result

    # lowers nested mapjoin logic IR nodes into a single pointwise expression
    def lower_to_pointwise(
        self,
        ex: LogicNode,
        einsums: list[Einsum],
        parameters: dict[str, Table],
        definitions: dict[str, Einsum],
    ) -> PointwiseNode:
        match ex:
            case Reorder(arg, idxs):
                return self.lower_to_pointwise(arg, einsums, parameters, definitions)
            case MapJoin(Literal(operation), args):
                args_list = [
                    self.lower_to_pointwise(arg, einsums, parameters, definitions)
                    for arg in args
                ]
                return self.lower_to_pointwise_op(operation, tuple(args_list))
            case Relabel(
                Alias(name), idxs
            ):  # relable is really just a glorified pointwise access
                return PointwiseAccess(
                    alias=PointwiseNamedField(name),
                    idxs=tuple(PointwiseNamedField(idx.name) for idx in idxs),
                )
            case Literal(value):
                return PointwiseLiteral(val=value)
            case Aggregate(
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
                return PointwiseAccess(
                    alias=PointwiseNamedField(aggregate_einsum_alias),
                    idxs=tuple(PointwiseNamedField(field.name) for field in ex.fields),
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

    def print_pointwise_op(self, pointwise_op: PointwiseOp):
        opstr = f" {self.print_pointwise_op_callable(pointwise_op.op)} "
        if not is_commutative(pointwise_op.op):
            return f"({pointwise_op.args[0]}{opstr}{pointwise_op.args[1]})"

        return f"({opstr.join(self.print_pointwise(arg) for arg in pointwise_op.args)})"

    def print_indicies(self, idxs: tuple[PointwiseNode, ...]):
        return ", ".join([self.print_pointwise(idx) for idx in idxs])

    def print_pointwise(self, pointwise_expr: PointwiseNode):
        match pointwise_expr:
            case Einsum(_, _, _, _):
                return self.print_einsum(pointwise_expr)
            case PointwiseNamedField(name):
                return name
            case PointwiseAccess(alias, idxs):
                return f"{self.print_pointwise(alias)}[{self.print_indicies(idxs)}]"
            case GetSparseCoordArray(sparse_tensor, dimension):
                return (
                    f"{self.print_pointwise(sparse_tensor)}Coords\
                    {self.print_pointwise(dimension)}"
                    if dimension
                    else f"{self.print_pointwise(sparse_tensor)}Coords"
                )
            case GetSparseValueArray(sparse_tensor):
                return f"{self.print_pointwise(sparse_tensor)}Values"
            case PointwiseOp(_, __):
                return self.print_pointwise_op(pointwise_expr)
            case PointwiseLiteral(val):
                return str(val)
            case PointwiseIfElse(condition, then_expr, else_expr):
                return f"ifelse({self.print_pointwise(condition)}, \
                    {self.print_pointwise(then_expr)}, \
                    {self.print_pointwise(else_expr)})"

    def print_einsum(self, einsum: Einsum) -> str:
        return (
            f"{self.print_pointwise(einsum.output)}["
            f"{self.print_indicies(einsum.output_fields)}] "
            f"{self.print_reducer(einsum.reduceOp)} "
            f"{self.print_pointwise(einsum.pointwise_expr)}"
        )

    def print_einsum_plan(self, einsum_plan: EinsumPlan) -> str:
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

    def __call__(self, prgm: EinsumPlan) -> str:
        return self.print_einsum_plan(prgm)


class EinsumInterpreter:
    def __call__(self, einsum_plan: EinsumPlan, parameters: dict[str, Table]):
        return self.print(einsum_plan, parameters)

    def print(self, einsum_plan: EinsumPlan, parameters: dict[str, Table]):
        for str, table in parameters.items():
            print(f"Parameter: {str} = {table}")

        print(einsum_plan)
        return (np.arange(6, dtype=np.float32).reshape(2, 3),)


class EinsumCompiler:
    def __init__(self):
        self.el = EinsumLowerer()

    def __call__(self, prgm: Plan):
        einsum_plan, parameters = self.el(prgm)

        return einsum_plan, parameters


class EinsumScheduler:
    def __init__(self, ctx: EinsumCompiler):
        self.ctx = ctx
        self.interpret = EinsumInterpreter()

    def __call__(self, prgm: LogicNode):
        if not isinstance(prgm, Plan):
            raise TypeError(f"EinsumScheduler expects a Plan, got {type(prgm)}")
        einsum_plan, parameters = self.ctx(prgm)
        return self.interpret(einsum_plan, parameters)
