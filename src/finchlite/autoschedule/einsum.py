from dataclasses import dataclass
from abc import ABC
from typing import Callable, Self

from finchlite.finch_logic import LogicNode, Field, Plan, Query, Alias, Literal, Relabel
from finchlite.finch_logic.nodes import Aggregate, MapJoin, Produces, Reorder
from finchlite.symbolic import Term, TermTree
from finchlite.autoschedule import optimize
from finchlite.algebra import is_commutative, overwrite, init_value

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

    @classmethod
    def rename(self, new_alias: str):
        return Einsum(self.reduceOp, self.input_fields, self.output_fields, self.pointwise_expr, new_alias)

    @classmethod
    def reorder(self, idxs: tuple[Field, ...]):
        return Einsum(self.reduceOp, idxs, self.output_fields, self.pointwise_expr, self.output_alias)

@dataclass(eq=True, frozen=True)
class EinsumPlan(Plan):
    """
    EinsumPlan
    
    A plan that contains einsum operations. Basically a list of einsum operations.
    """

    bodies: tuple[Einsum, ...]
    returnValue: Einsum | None

    @classmethod
    def from_children(cls, bodies: tuple[Einsum, ...], returnValue: Einsum | None) -> Self:
        return cls(bodies, returnValue)

    @property
    def children(self) -> tuple[Einsum, ...]:
        return [*self.bodies, self.returnValue]

class EinsumLowerer:
    alias_counter: int = 0

    def __call__(self, prgm: Plan) -> EinsumPlan:
        return self.compile_plan(prgm)

    def get_next_alias(self) -> str:
        self.alias_counter += 1
        return f"einsum_{self.alias_counter}"

    def compile_plan(self, plan: Plan) -> EinsumPlan:
        einsums = []
        returnValue = None

        for body in plan.bodies:
            match body:
                case Plan(_):
                    plan = self.compile_plan(body)
                    if plan.returnValue is not None:
                        raise Exception("Plans with return values are not statements, but rather are expressions.")
                    einsums.extend(plan.bodies)
                case Query(Alias(name), rhs):
                    einsums.append(self.lower_to_einsum(rhs, einsums).rename(name))
                case Produces(arg):
                    if returnValue is not None:
                        raise Exception("Only one return value is supported.")
                    returnValue = self.lower_to_einsum(arg, einsums)
                case _:
                    einsums.append(self.lower_to_einsum(body, einsums).rename(self.get_next_alias()))
        
        return EinsumPlan(tuple(einsums), returnValue)

    def lower_to_einsum(self, ex: LogicNode, einsums: list[Einsum]) -> Einsum:
        match ex:
            case Plan(_):
                plan = self.compile_plan(ex)
                einsums.extend(plan.bodies)
                return plan.returnValue 
            case MapJoin(Literal(operation), args):
                args = [self.lower_to_pointwise(arg, einsums) for arg in args]
                pointwise_expr = self.lower_to_pointwise_op(operation, args)
                return Einsum(reduceOp=overwrite, input_fields=ex.fields, output_fields=ex.fields, pointwise_expr=pointwise_expr, output_alias=None)
            case Reorder(arg, idxs):
                return self.lower_to_einsum(arg, einsums).reorder(idxs)
            case Aggregate(operation, init, arg, idxs):
                if init != init_value(operation, type(init)):
                    raise Exception(f"Init value {init} is not the default value for operation {operation} of type {type(init)}. Non standard init values are not supported.")
                pointwise_expr = self.lower_to_pointwise(arg, einsums)
                return Einsum(operation, arg.fields, ex.fields, pointwise_expr, self.get_next_alias(), output_alias=None)
            case _:
                raise Exception(f"Unrecognized logic: {ex}")

    def lower_to_pointwise_op(self, operation: Callable, args: tuple[PointwiseNode, ...]) -> PointwiseOp:
        # if operation is commutative, we simply pass all the args to the pointwise op since order of args does not matter
        if is_commutative(operation):
            args = [] # flatten the args
            for arg in args:
                match arg:
                    case PointwiseOp(op2, _) if op2 == operation:
                        args.extend(arg.args)
                    case _:
                        args.append(arg)

            return PointwiseOp(operation, args)

        # combine args from left to right (i.e a / b / c -> (a / b) / c)
        assert len(args) > 1
        result = PointwiseOp(operation, args[0], args[1])
        for arg in args[2:]:
            result = PointwiseOp(operation, result, arg)
        return result

    # lowers nested mapjoin logic IR nodes into a single pointwise expression
    def lower_to_pointwise(self, ex: LogicNode, einsums: list[Einsum]) -> PointwiseNode:
        match ex:
            case MapJoin(Literal(operation), args):
                args = [self.lower_to_pointwise(arg, einsums) for arg in args]
                return self.lower_to_pointwise_op(operation, args)
            case Relabel(Alias(name), idxs): # relable is really just a glorified pointwise access
                return PointwiseAccess(alias=name, idxs=idxs)
            case Literal(value):
                return PointwiseLiteral(val=value)
            case Aggregate(_, _, _, _): # aggregate has to be computed seperatley as it's own einsum
                aggregate_einsum_alias = self.get_next_alias()
                einsums.append(self.lower_to_einsum(ex, einsums).rename(aggregate_einsum_alias)) 
                return PointwiseAccess(alias=aggregate_einsum_alias, idxs=tuple(ex.fields))
            case _:
                raise Exception(f"Unrecognized logic: {ex}")

class EinsumCompiler:
    def __init__(self):
        self.el = EinsumLowerer()

    def __call__(self, prgm: Plan):
        return self.el(prgm)

def einsum_scheduler(plan: Plan):
    optimized_prgm = optimize(plan)

    interpreter = EinsumCompiler()
    return interpreter(optimized_prgm)