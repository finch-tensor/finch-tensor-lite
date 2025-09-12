from dataclasses import dataclass
from abc import ABC
from typing import Callable, Self

from finchlite.finch_logic import LogicExpression, LogicNode, Field, Plan
from finchlite.symbolic import Term, TermTree
from finchlite.autoschedule import optimize


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

    tensor: LogicExpression
    idxs: tuple[Field, ...]  # (Field('i'), Field('j'))
    # Children: None (leaf)

    @classmethod
    def from_children(cls, tensor: LogicExpression, idxs: tuple[Field, ...]) -> Self:
        return cls(tensor, idxs)

    @property
    def children(self):
        return [self.tensor, *self.idxs]

@dataclass(eq=True, frozen=True)
class PointwiseOp(PointwiseNode):
    """
    PointwiseOp

    Represents an operation like + or * on pointwise expressions for multiple operands.
    If operation is not commutative, pointwise node must be binary, with 2 args at most.

    Attributes:
        op: The function to apply e.g., operator.add
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

    @classmethod
    def from_children(cls, val: float) -> Self:
        return cls(val)

    @property
    def children(self):
        return [self.val]

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

    updateOp: Callable

    input_fields = tuple[Field, ...]    # indicies that are used in the pointwise expression (i.e. i, j, k)
    output_fields = tuple[Field, ...]   # a subset of input_fields that are used in the output (i.e. i, j)
    pointwise_expr: PointwiseNode       # the pointwise expression that is aggregated
    
    @classmethod
    def from_children(cls, updateOp: Callable, input_fields: tuple[Field, ...], output_fields: tuple[Field, ...], pointwise_expr: PointwiseNode) -> Self:
        return cls(updateOp, input_fields, output_fields, pointwise_expr)
    
    @property
    def children(self):
        return [self.updateOp, self.input_fields, self.output_fields, self.pointwise_expr]

@dataclass(eq=True, frozen=True)
class EinsumPlan(Plan):
    """
    EinsumPlan
    
    A plan that contains einsum operations. Basically a list of einsum operations.
    """

    bodies: tuple[Einsum, ...]

    @classmethod
    def from_children(cls, bodies: tuple[Einsum, ...]) -> Self:
        return cls(bodies)

    @property
    def children(self) -> tuple[Einsum, ...]:
        return self.bodies

def make_einsum_plan(bodies: tuple[Einsum | EinsumPlan, ...]) -> EinsumPlan:
    """Flatten nested EinsumPlans so the resulting tuple contains only Einsum nodes."""
    flat: list[Einsum] = []
    for body in bodies:
        if isinstance(body, EinsumPlan):
            flat.extend(body.children)
        else:
            flat.append(body)
    return EinsumPlan(tuple(flat))

class EinsumLowerer:
    def __call__(self, ex: LogicNode) -> EinsumPlan:
        match ex:
            case Plan(bodies):
                return make_einsum_plan(tuple(self(body) for body in bodies))
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