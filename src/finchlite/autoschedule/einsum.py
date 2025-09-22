from dataclasses import dataclass
from abc import ABC
import operator
from typing import Callable, Self

from finchlite.finch_logic import LogicNode, Field, Plan, Query, Alias, Literal, Relabel
from finchlite.finch_logic.nodes import Aggregate, MapJoin, Produces, Reorder, Table
from finchlite.symbolic import Term, TermTree
from finchlite.autoschedule import optimize
from finchlite.algebra import is_commutative, overwrite, init_value, promote_max, promote_min

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
        return ctx.print_pointwise_expr(self)

@dataclass(eq=True, frozen=True)
class PointwiseAccess(PointwiseNode, TermTree):
    """
    PointwiseAccess

    Tensor access like a[i, j].

    Attributes:
        tensor: The tensor to access.
        idxs: The indices at which to access the tensor.
    """

    alias: str
    idxs: tuple[Field, ...]  # (Field('i'), Field('j'))
    # Children: None (leaf)

    @classmethod
    def from_children(cls, alias: str, idxs: tuple[Field, ...]) -> Self:
        return cls(alias, idxs)

    @property
    def children(self):
        return [self.alias, *self.idxs]

@dataclass(eq=True, frozen=True)
class PointwiseOp(PointwiseNode):
    """
    PointwiseOp

    Represents an operation like + or * on pointwise expressions for multiple operands.
    If operation is not commutative, pointwise node must be binary, with 2 args at most.

    Attributes:
        op: The function to apply e.g., operator.add, operator.mul, operator.subtract, operator.div, etc... Must be a callable.
        args: The arguments to the operation.
    """

    op: Callable  #the function to apply e.g., operator.add
    args: tuple[PointwiseNode, ...]  # Subtrees
    # Children: The args

    @classmethod
    def from_children(cls, op: Callable, args: tuple[PointwiseNode, ...]) -> Self:
        return cls(op, args)

    @property
    def children(self):
        return [self.op, *self.args]

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


#einsum and einsum ast not part of logic IR
#transform to it's own language
@dataclass(eq=True, frozen=True)
class Einsum(TermTree):
    """
    Einsum

    A einsum operation that maps pointwise expressions and aggregates them.

    Attributes:
        updateOp: The function to apply to the pointwise expressions (e.g. +=, f=, max=, etc...).
        input_fields: The indices that are used in the pointwise expression (i.e. i, j, k).
        output_fields: The indices that are used in the output (i.e. i, j).
        pointwise_expr: The pointwise expression that is mapped and aggregated.
    """

    reduceOp: Callable #technically a reduce operation, much akin to the one in aggregate

    input_fields: tuple[Field, ...]
    output_fields: tuple[Field, ...]
    pointwise_expr: PointwiseNode
    output_alias: str | None
    
    @classmethod
    def from_children(cls, output_alias: str | None, updateOp: Callable, input_fields: tuple[Field, ...], output_fields: tuple[Field, ...], pointwise_expr: PointwiseNode) -> Self:
        return cls(output_alias, updateOp, input_fields, output_fields, pointwise_expr)
    
    @property
    def children(self):
        return [self.output_alias, self.reduceOp, self.input_fields, self.output_fields, self.pointwise_expr]

    def rename(self, new_alias: str):
        return Einsum(self.reduceOp, self.input_fields, self.output_fields, self.pointwise_expr, new_alias)

    def reorder(self, idxs: tuple[Field, ...]):
        return Einsum(self.reduceOp, idxs, self.output_fields, self.pointwise_expr, self.output_alias)

    def __str__(self):
        ctx = EinsumPrinterContext()
        return ctx.print_einsum(self)

@dataclass(eq=True, frozen=True)
class EinsumPlan(Plan):
    """
    EinsumPlan
    
    A plan that contains einsum operations. Basically a list of einsum operations.
    """

    bodies: tuple[Einsum, ...] = ()
    returnValues: tuple[Einsum | str] = ()

    @classmethod
    def from_children(cls, bodies: tuple[Einsum, ...], returnValue: tuple[Einsum | str]) -> Self:
        return cls(bodies, returnValue)

    @property
    def children(self):
        return [*self.bodies, self.returnValues]

    def __str__(self):
        ctx = EinsumPrinterContext()
        return ctx(self)

class EinsumLowerer:
    alias_counter: int = 0

    def __call__(self, prgm: Plan, parameters: dict[str, Table], definitions: dict[str, Einsum]) -> EinsumPlan:
        return self.compile_plan(prgm, parameters, definitions)

    def get_next_alias(self) -> str:
        self.alias_counter += 1
        return f"einsum_{self.alias_counter}"

    def rename_einsum(self, einsum: Einsum, new_alias: str, definitions: dict[str, Einsum]) -> Einsum:
        definitions[new_alias] = einsum
        return einsum.rename(new_alias)

    def compile_plan(self, plan: Plan, parameters: dict[str, Table], definitions: dict[str, Einsum]) -> EinsumPlan:
        einsums = []
        returnValue = []

        for body in plan.bodies:
            match body:
                case Plan(_):
                    einsum_plan = self.compile_plan(body, parameters, definitions)
                    einsums.extend(einsum_plan.bodies)

                    if einsum_plan.returnValues:
                        if returnValue:
                            raise Exception("Cannot invoke return more than once.")
                        returnValue = einsum_plan.returnValues
                case Query(Alias(name), Table(_, _)):
                    parameters[name] = body.rhs
                case Query(Alias(name), rhs):
                    einsums.append(self.rename_einsum(self.lower_to_einsum(rhs, einsums, parameters, definitions), name, definitions))
                case Produces(args):
                    if returnValue:
                        raise Exception("Cannot invoke return more than once.")
                    for arg in args:
                        returnValue.append(arg.name if isinstance(arg, Alias) else self.lower_to_einsum(arg, einsums, parameters, definitions))
                case _:
                    einsums.append(self.rename_einsum(self.lower_to_einsum(body, einsums, parameters, definitions), self.get_next_alias(), definitions))
        
        return EinsumPlan(tuple(einsums), tuple(returnValue))

    def lower_to_einsum(self, ex: LogicNode, einsums: list[Einsum], parameters: dict[str, Table], definitions: dict[str, Einsum]) -> Einsum:
        match ex:
            case Plan(_):
                plan = self.compile_plan(ex, parameters, definitions)
                einsums.extend(plan.bodies)
                
                if plan.returnValues:
                    raise Exception("Plans with no return value are not statements, but rather are expressions.")
                
                if len(plan.returnValues) > 1:
                    raise Exception("Only one return value is supported.")

                if isinstance(plan.returnValues[0], str):
                    returned_alias = plan.returnValues[0]
                    returned_einsum = definitions[returned_alias]
                    return PointwiseAccess(alias=returned_alias, idxs=returned_einsum.output_fields)
                
                return plan.returnValues[0] 
            case MapJoin(Literal(operation), args):
                args = [self.lower_to_pointwise(arg, einsums, parameters, definitions) for arg in args]
                pointwise_expr = self.lower_to_pointwise_op(operation, args)
                return Einsum(reduceOp=overwrite, input_fields=ex.fields, output_fields=ex.fields, pointwise_expr=pointwise_expr, output_alias=None)
            case Reorder(arg, idxs):
                return self.lower_to_einsum(arg, einsums, parameters, definitions).reorder(idxs)
            case Aggregate(Literal(operation), Literal(init), arg, idxs):
                if init != init_value(operation, type(init)):
                    raise Exception(f"Init value {init} is not the default value for operation {operation} of type {type(init)}. Non standard init values are not supported.")
                pointwise_expr = self.lower_to_pointwise(arg, einsums, parameters, definitions)
                return Einsum(operation, arg.fields, ex.fields, pointwise_expr, self.get_next_alias())
            case _:
                raise Exception(f"Unrecognized logic: {ex}")

    def lower_to_pointwise_op(self, operation: Callable, args: tuple[PointwiseNode, ...]) -> PointwiseOp:
        # if operation is commutative, we simply pass all the args to the pointwise op since order of args does not matter
        if is_commutative(operation):
            ret_args = [] # flatten the args
            for arg in args:
                match arg:
                    case PointwiseOp(op2, _) if op2 == operation:
                        ret_args.extend(arg.args)
                    case _:
                        ret_args.append(arg)

            return PointwiseOp(operation, ret_args)

        # combine args from left to right (i.e a / b / c -> (a / b) / c)
        assert len(args) > 1
        result = PointwiseOp(operation, args[0], args[1])
        for arg in args[2:]:
            result = PointwiseOp(operation, result, arg)
        return result

    # lowers nested mapjoin logic IR nodes into a single pointwise expression
    def lower_to_pointwise(self, ex: LogicNode, einsums: list[Einsum], parameters: dict[str, Table], definitions: dict[str, Einsum]) -> PointwiseNode:
        match ex:
            case Reorder(arg, idxs):
                return self.lower_to_pointwise(arg, einsums, parameters, definitions)
            case MapJoin(Literal(operation), args):
                args = [self.lower_to_pointwise(arg, einsums, parameters, definitions) for arg in args]
                return self.lower_to_pointwise_op(operation, args)
            case Relabel(Alias(name), idxs): # relable is really just a glorified pointwise access
                return PointwiseAccess(alias=name, idxs=idxs)
            case Literal(value):
                return PointwiseLiteral(val=value)
            case Aggregate(_, _, _, _): # aggregate has to be computed seperatley as it's own einsum
                aggregate_einsum_alias = self.get_next_alias()
                einsums.append(self.rename_einsum(self.lower_to_einsum(ex, einsums, parameters, definitions), aggregate_einsum_alias, definitions)) 
                return PointwiseAccess(alias=aggregate_einsum_alias, idxs=tuple(ex.fields))
            case _:
                raise Exception(f"Unrecognized logic: {ex}")

class EinsumCompiler:
    def __init__(self):
        self.el = EinsumLowerer()

    def __call__(self, prgm: Plan):
        parameters = {}
        definitions = {}
        return self.el(prgm, parameters, definitions), parameters, definitions

def einsum_scheduler(plan: Plan):
    optimized_prgm = optimize(plan)

    compiler = EinsumCompiler()
    return compiler(optimized_prgm)

class EinsumPrinterContext:
    def print_indicies(self, idxs: tuple[Field, ...]):
        return ", ".join([str(idx) for idx in idxs])
    
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
        if is_commutative(pointwise_op.op) == False:
            return f"({pointwise_op.args[0]} {self.print_pointwise_op_callable(pointwise_op.op)} {pointwise_op.args[1]})"
        return f"({f" {self.print_pointwise_op_callable(pointwise_op.op)} ".join(self.print_pointwise_expr(arg) for arg in pointwise_op.args)})"

    def print_pointwise_expr(self, pointwise_expr: PointwiseNode):
        match pointwise_expr:
            case PointwiseAccess(alias, idxs):
                return f"{alias}[{self.print_indicies(idxs)}]"
            case PointwiseOp(_, __):
                return self.print_pointwise_op(pointwise_expr)
            case PointwiseLiteral(val):
                return str(val)

    def print_einsum(self, einsum: Einsum):
        return f"{einsum.output_alias}[{self.print_indicies(einsum.output_fields)}] {self.print_reducer(einsum.reduceOp)} {self.print_pointwise_expr(einsum.pointwise_expr)}"
    
    def print_return_value(self, return_value: Einsum | str):
        return return_value if isinstance(return_value, str) else self.print_einsum(return_value)

    def print_einsum_plan(self, einsum_plan: EinsumPlan):
        if not einsum_plan.returnValues:
            return "\n".join([self.print_einsum(einsum) for einsum in einsum_plan.bodies])
        return f"{"\n".join([self.print_einsum(einsum) for einsum in einsum_plan.bodies])}\nreturn {", ".join([self.print_return_value(return_value) for return_value in einsum_plan.returnValues])}"
    
    def __call__(self, prgm: EinsumPlan):
        return self.print_einsum_plan(prgm)