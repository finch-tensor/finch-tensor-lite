from ast import Tuple
from dataclasses import dataclass
from abc import ABC
import operator
from turtle import st
from typing import Callable, Self

from finchlite.algebra.tensor import Tensor
from finchlite.finch_logic import LogicNode, Field, Plan, Query, Alias, Literal, Relabel
from finchlite.finch_logic.nodes import Aggregate, MapJoin, Produces, Reorder, Table
from finchlite.symbolic import Term, TermTree
from finchlite.algebra import is_commutative, overwrite, init_value, promote_max, promote_min
import numpy as np

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
class PointwiseIndirectCOOAccess(PointwiseNode, TermTree):
    """
    PointwiseIndirectCOOAccess

    Tensor access like a[i, j] but for sparse tensors. So in reality it's like a[COO_coords[i]] = ...

    Attributes:
        tensor: The tensor to access.
        coo_coords: The COO coordinates at which to access the tensor (this is also a tensor).
        idxs: The indices at which to access the tensor.
    """

    alias: str
    coo_coord_alias: str
    idx: Field #only one index is needed to access the COO coord tensor
    # Children: None (leaf)

    @classmethod
    def from_children(cls, alias: str, coo_coord_alias: str, idx: Field) -> Self:
        return cls(alias, coo_coord_alias, idx)

    @property
    def children(self):
        return [self.alias, self.coo_coord_alias, self.idx]

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
    #input_fields: tuple[tuple[Field, ...], ...] 
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

@dataclass(eq=True, frozen=True)
class EinsumPlanStatement(Term, ABC):
    """
    EinsumPlanStatement

    Represents an AST node in the Einsum Plan IR
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
        return ctx.print_einsum_plan_statement(self)

#einsum and einsum ast not part of logic IR
#transform to it's own language
@dataclass(eq=True, frozen=True)
class Einsum(EinsumPlanStatement, TermTree):
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

    #input_fields: tuple[Field, ...] #redundant remove later
    output_fields: tuple[Field, ...]
    pointwise_expr: PointwiseNode

    output_alias: str | None
    indirect_coo_alias: str | None
    
    @classmethod
    def from_children(cls, output_alias: str | None, updateOp: Callable, output_fields: tuple[Field, ...], pointwise_expr: PointwiseNode, indirect_coo_alias: str | None) -> Self:
        #return cls(output_alias, updateOp, input_fields, output_fields, pointwise_expr)
        return cls(output_alias, updateOp, output_fields, pointwise_expr, indirect_coo_alias)
    
    @property
    def children(self):
        #return [self.output_alias, self.reduceOp, self.input_fields, self.output_fields, self.pointwise_expr]
        return [self.output_alias, self.reduceOp, self.output_fields, self.pointwise_expr, self.indirect_coo_alias]

    def rename(self, new_alias: str):
        #return Einsum(self.reduceOp, self.input_fields, self.output_fields, self.pointwise_expr, new_alias)
        return Einsum(self.reduceOp, self.output_fields, self.pointwise_expr, new_alias, self.indirect_coo_alias)

    def reorder(self, idxs: tuple[Field, ...]):
        #return Einsum(self.reduceOp, idxs, self.output_fields, self.pointwise_expr, self.output_alias)
        return Einsum(self.reduceOp, idxs, self.pointwise_expr, self.output_alias, self.indirect_coo_alias)

@dataclass(eq=True, frozen=True)
class ExtractCOO(EinsumPlanStatement):
    """
    ExtractCOO
    
    A plan statement that contains an extract's the COO matrix from a sparse tensor.
    """
    alias: str

    @classmethod
    def from_children(cls, alias: str) -> Self:
        return cls(alias)

    @property
    def children(self):
        return [self.alias]

@dataclass(eq=True, frozen=True)
class EinsumPlan(Plan):
    """
    EinsumPlan
    
    A plan that contains einsum operations. Basically a list of einsum operations.
    """

    bodies: tuple[EinsumPlanStatement, ...] = ()
    returnValues: tuple[Einsum | str] = ()

    @classmethod
    def from_children(cls, bodies: tuple[EinsumPlanStatement, ...], returnValue: tuple[Einsum | str]) -> Self:
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
        einsum_statements: list[EinsumPlanStatement] = []
        returnValue = []

        for body in plan.bodies:
            match body:
                case Plan(_):
                    einsum_plan = self.compile_plan(body, parameters, definitions)
                    einsum_statements.extend(einsum_plan.bodies)

                    if einsum_plan.returnValues:
                        if returnValue:
                            raise Exception("Cannot invoke return more than once.")
                        returnValue = einsum_plan.returnValues
                case Query(Alias(name), Table(_, _)):
                    parameters[name] = body.rhs
                case Query(Alias(name), rhs):
                    einsum_statements.append(self.rename_einsum(self.lower_to_einsum(rhs, einsum_statements, parameters, definitions), name, definitions))
                case Produces(args):
                    if returnValue:
                        raise Exception("Cannot invoke return more than once.")
                    for arg in args:
                        returnValue.append(arg.name if isinstance(arg, Alias) else self.lower_to_einsum(arg, einsum_statements, parameters, definitions))
                case _:
                    einsum_statements.append(self.rename_einsum(self.lower_to_einsum(body, einsum_statements, parameters, definitions), self.get_next_alias(), definitions))
        
        return EinsumPlan(tuple(einsum_statements), tuple(returnValue))

    def lower_to_einsum(self, ex: LogicNode, einsum_statements: list[EinsumPlanStatement], parameters: dict[str, Table], definitions: dict[str, Einsum]) -> Einsum:
        match ex:
            case Plan(_):
                plan = self.compile_plan(ex, parameters, definitions)
                einsum_statements.extend(plan.bodies)
                
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
                args = [self.lower_to_pointwise(arg, einsum_statements, parameters, definitions) for arg in args]
                pointwise_expr = self.lower_to_pointwise_op(operation, args)
                #return Einsum(reduceOp=overwrite, input_fields=ex.fields, output_fields=ex.fields, pointwise_expr=pointwise_expr, output_alias=None)
                return Einsum(reduceOp=overwrite, output_fields=ex.fields, pointwise_expr=pointwise_expr, output_alias=None)
            case Reorder(arg, idxs):
                return self.lower_to_einsum(arg, einsum_statements, parameters, definitions).reorder(idxs)
            case Aggregate(Literal(operation), Literal(init), arg, idxs):
                if init != init_value(operation, type(init)):
                    raise Exception(f"Init value {init} is not the default value for operation {operation} of type {type(init)}. Non standard init values are not supported.")
                pointwise_expr = self.lower_to_pointwise(arg, einsum_statements, parameters, definitions)
                #return Einsum(operation, arg.fields, ex.fields, pointwise_expr, self.get_next_alias())
                return Einsum(operation, ex.fields, pointwise_expr, self.get_next_alias(), None)
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
    def lower_to_pointwise(self, ex: LogicNode, einsum_statements: list[EinsumPlanStatement], parameters: dict[str, Table], definitions: dict[str, Einsum]) -> PointwiseNode:
        match ex:
            case Reorder(arg, idxs):
                return self.lower_to_pointwise(arg, einsum_statements, parameters, definitions)
            case MapJoin(Literal(operation), args):
                args = [self.lower_to_pointwise(arg, einsum_statements, parameters, definitions) for arg in args]
                return self.lower_to_pointwise_op(operation, args)
            case Relabel(Alias(name), idxs): # relable is really just a glorified pointwise access
                return PointwiseAccess(alias=name, idxs=idxs)
            case Literal(value):
                return PointwiseLiteral(val=value)
            case Aggregate(_, _, _, _): # aggregate has to be computed seperatley as it's own einsum
                aggregate_einsum_alias = self.get_next_alias()
                einsum_statements.append(self.rename_einsum(self.lower_to_einsum(ex, einsum_statements, parameters, definitions), aggregate_einsum_alias, definitions)) 
                return PointwiseAccess(alias=aggregate_einsum_alias, idxs=tuple(ex.fields))
            case _:
                raise Exception(f"Unrecognized logic: {ex}")

class EinsumCompiler:
    def __init__(self):
        self.el = EinsumLowerer()

    def find_sparse_tensors(self, parameters: dict[str, Table])-> dict: # -> dict[str, Tuple[Field, ...]]: getting type errors here
        from finchlite.autoschedule.sparse_tensor import SparseTensor
        
        sparse_tensors = dict()
        for alias, value in parameters.items():
            match value:
                case Table(tensor, idxs):
                    if isinstance(tensor, SparseTensor):
                        sparse_tensors[alias] = idxs
        return sparse_tensors

    #getting type errors here if I use dict[str, Tuple[Field, ...]]
    def optimize_einsum(self, einsum_plan: EinsumPlan, sparse_aliases: dict) -> EinsumPlan:
        def optimize_sparse_einsum(einsum: Einsum, extra_ops: list[EinsumPlanStatement]) -> Einsum:
            return einsum

        optimized_einsums: list[EinsumPlanStatement] = []
        for statement in einsum_plan.bodies:
            match statement:
                case Einsum(_, _, _, _, _):
                    optimized_einsums.append(optimize_sparse_einsum(statement, optimized_einsums))
                case _:
                    optimized_einsums.append(statement)

        optimized_returns = []
        for return_value in einsum_plan.returnValues:
            match return_value:
                case Einsum(_, _, _, _, _):
                    optimized_returns.append(optimize_sparse_einsum(return_value, optimized_einsums))
                case _:
                    optimized_returns.append(return_value)
        return EinsumPlan(tuple(optimized_einsums), tuple(optimized_returns))

    def __call__(self, prgm: Plan):
        parameters = {}
        definitions = {}
        einsum_plan = self.el(prgm, parameters, definitions)

        sparse_aliases = self.find_sparse_tensors(parameters)
        einsum_plan = self.optimize_einsum(einsum_plan, sparse_aliases)

        return einsum_plan, parameters, definitions

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
            case PointwiseIndirectCOOAccess(alias, coo_coord_alias, idx):
                return f"{alias}[{coo_coord_alias}[{self.print_indicies((idx, ))}]]"
            case PointwiseOp(_, __):
                return self.print_pointwise_op(pointwise_expr)
            case PointwiseLiteral(val):
                return str(val)

    def print_einsum(self, einsum: Einsum) -> str:
        if einsum.indirect_coo_alias:
            return f"{einsum.output_alias}[{einsum.indirect_coo_alias}[{self.print_indicies(einsum.output_fields)}]] {self.print_reducer(einsum.reduceOp)} {self.print_pointwise_expr(einsum.pointwise_expr)}"
        return f"{einsum.output_alias}[{self.print_indicies(einsum.output_fields)}] {self.print_reducer(einsum.reduceOp)} {self.print_pointwise_expr(einsum.pointwise_expr)}"
    
    def print_return_value(self, return_value: Einsum | str) -> str:
        return return_value if isinstance(return_value, str) else self.print_einsum(return_value)

    def print_einsum_plan_statement(self, einsum_plan_statement: EinsumPlanStatement) -> str:
        match einsum_plan_statement:
            case Einsum(_, _, _, _, _):
                return self.print_einsum(einsum_plan_statement)
            case _:
                raise Exception(f"Unrecognized einsum plan statement: {einsum_plan_statement}")

    def print_einsum_plan(self, einsum_plan: EinsumPlan) -> str:
        if not einsum_plan.returnValues:
            return "\n".join([self.print_einsum(einsum) for einsum in einsum_plan.bodies])
        return f"{"\n".join([self.print_einsum_plan_statement(statement) for statement in einsum_plan.bodies])}\nreturn {", ".join([self.print_return_value(return_value) for return_value in einsum_plan.returnValues])}"
    
    def __call__(self, prgm: EinsumPlan) -> str:
        return self.print_einsum_plan(prgm)

class EinsumInterpreter:
    def __call__(self, einsum_plan: EinsumPlan, parameters: dict[str, Table]):
        return self.print(einsum_plan, parameters)

    def print(self, einsum_plan: EinsumPlan, parameters: dict[str, Table]):
        for (str, table) in parameters.items():
            print(f"Parameter: {str} = {table}")
        
        print(einsum_plan)
        return (np.arange(6, dtype=np.float32).reshape(2, 3),)

class EinsumScheduler:
    def __init__(self, ctx: EinsumCompiler):
        self.ctx = ctx
        self.interpret = EinsumInterpreter()

    def __call__(self, prgm: LogicNode):
        einsum_plan, parameters, _ = self.ctx(prgm)
        return self.interpret(einsum_plan, parameters)