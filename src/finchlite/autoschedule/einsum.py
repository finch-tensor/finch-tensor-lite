from dataclasses import dataclass
from abc import ABC
from typing import Callable, Self
import operator

from finchlite.finch_logic import LogicExpression, MapJoin, Aggregate, LogicNode, Alias, Table, Literal, Field, Relabel, Reorder
from finchlite.autoschedule import DefaultLogicOptimizer, LogicCompiler, optimize
from finchlite.symbolic import Rewrite, PostWalk, PostOrderDFS, Term, TermTree


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
    """Tensor access like a[i, j]."""
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
    """Operation like + or *."""
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
class Einsum:
    """
    NumPy-style einsum logic node.

    - inputs: per-argument axis labels as Fields, e.g., ((i,k), (k,j))
    - output: output axis labels as Fields, e.g., (i,j)
    - args:   input expressions
    """

    isEinProduct: bool

    inputs: tuple[tuple[Field, ...], ...]
    outputs: tuple[Field, ...]
    args: tuple[LogicExpression, ...]

    @property
    def children(self):
        # Treat only args as children in the term tree
        return list(self.args)

    @property
    def fields(self) -> list[Field]:
        return list(self.outputs)

class EinsumTransformer(DefaultLogicOptimizer):
    """
    Rewrite unoptimized Logic IR (mostly MapJoin and Aggregate) into Einsum nodes.

    Pattern handled:
    - Aggregate(add, 0, MapJoin(mul, args), reduce_idxs) -> Einsum(args, outputs=reduce_idxs)
    - Aggregate(add, 0, Relabel(MapJoin(mul, args), _), reduce_idxs)-> Einsum(args, outputs=reduce_idxs)
    - Aggregate(add, 0, Reorder(MapJoin(mul, args), _), reduce_idxs)-> Einsum(args, outputs=reduce_idxs)
    - Aggregate(add, 0, Relabel(Reorder(MapJoin(mul, args), _), _), reduce_idxs)-> Einsum(args, outputs=reduce_idxs)
    - Aggregate(add, 0, Reorder(Relabel(MapJoin(mul, args), _), _), reduce_idxs)-> Einsum(args, outputs=reduce_idxs)
    - Aggregate(add, 0, Einsum(...), reduce_idxs)-> Einsum(..., outputs=reduce_idxs)

    - MapJoin(mul, args)-> Einsum(args) #elementwise product (no contraction)
    - Aggregate(mul, 0, MapJoin(mul, args), reduce_idxs)-> Einsum(args, outputs=reduce_idxs, isEinProduct=True) #elementwise product (no contraction)

    """

    def __init__(self, ctx: LogicCompiler, verbose=True):
        super().__init__(ctx)
        self.verbose = verbose
    
    def __call__(self, prgm: LogicNode):
        prgm = optimize(prgm)
        prgm = self.ctx(prgm)

        transformed = self.transform(prgm)

        return transformed

    def make_einsum(self, args: tuple[LogicExpression, ...], inputs: tuple[tuple[Field, ...], ...] | None = None, outputs: tuple[Field, ...] | None = None, isEinProduct: bool = False):
        #inputs are the fields of the arguments by default
        construct_inputs = inputs if inputs is not None else tuple(tuple(f for f in arg.fields) for arg in args)

        #outputs are the union of the inputs by default, or the union of the outputs if provided
        union_fields: list[Field] = list(outputs) if outputs is not None else list() #union fields are inputs by default
        for labels in construct_inputs:
            for f in labels:
                if f not in union_fields:
                    union_fields.append(f)
        construct_outputs = tuple(union_fields)
        return Einsum(args=args, inputs=construct_inputs, outputs=construct_outputs, isEinProduct=isEinProduct)

    def transform(self, prgm: LogicNode):
        def rule(node):
            match node:

                # Sum over product with harmless wrappers -> Einsum
                case Aggregate(Literal(operator.add), Literal(0), Relabel(MapJoin(Literal(operator.mul), args), _), idxs):
                    return self.make_einsum(args=args, inputs=None, outputs=idxs)
                case Aggregate(Literal(operator.add), Literal(0), Reorder(MapJoin(Literal(operator.mul), args), _), idxs):
                    return self.make_einsum(args=args, inputs=None, outputs=idxs)
                case Aggregate(
                    Literal(operator.add),
                    Literal(0),
                    Relabel(Reorder(MapJoin(Literal(operator.mul), args), _), _),
                    idxs,
                ):
                    return self.make_einsum(args=args, inputs=None, outputs=idxs)
                case Aggregate(
                    Literal(operator.add),
                    Literal(0),
                    Reorder(Relabel(MapJoin(Literal(operator.mul), args), _), _),
                    idxs,
                ):
                    return self.make_einsum(args=args, inputs=None, outputs=idxs)

                # Sum over already-converted Einsum (e.g., MapJoin->Einsum happened earlier)
                case Aggregate(Literal(operator.add), Literal(0), Einsum(args=args, inputs=_, outputs=_), idxs):
                    return self.make_einsum(args=args, inputs=None, outputs=idxs)

                # Original core pattern matching rules
                # Sum over product -> Einsum(no contraction)
                case Aggregate(Literal(operator.add), Literal(0), MapJoin(Literal(operator.mul), args), idxs):
                    return self.make_einsum(args=args, inputs=None, outputs=idxs)
                # Sum over product -> Einsum (no contraction)
                case Aggregate(Literal(operator.mul), Literal(0), MapJoin(Literal(operator.mul), args), idxs):
                    return self.make_einsum(args=args, inputs=None, outputs=idxs, isEinProduct=True)
                # Pure elementwise product -> Einsum (no contraction)
                case MapJoin(Literal(operator.mul), args):
                    return self.make_einsum(args=args, inputs=None, outputs=None)

        return Rewrite(PostWalk(rule))(prgm)


class PrintingLogicOptimizer(DefaultLogicOptimizer):
    """Custom optimizer that prints MapJoin and Aggregate operations"""
    
    def __init__(self, ctx: LogicCompiler, verbose=True):
        super().__init__(ctx)
        self.verbose = verbose
        self.operation_count = {"MapJoin": 0, "Aggregate": 0}
    
    def __call__(self, prgm: LogicNode):
        # First optimize the program
        prgm = optimize(prgm)
        
        # Then traverse and print all MapJoin/Aggregate operations
        if self.verbose:
            print("\n=== Finch Logic IR Operations ===")
            self._print_operations(prgm)
            print(f"\nTotal MapJoins: {self.operation_count['MapJoin']}")
            print(f"Total Aggregates: {self.operation_count['Aggregate']}")
            print("================================\n")
        
        # Continue with compilation
        return self.ctx(prgm)
    
    def _print_operations(self, node):
        """Traverse the Logic IR and print MapJoin/Aggregate operations"""
        for n in PostOrderDFS(node):
            match n:
                case MapJoin(op, args):
                    self.operation_count["MapJoin"] += 1
                    print(f"\nMapJoin #{self.operation_count['MapJoin']}:")
                    print(f"  Operation: {self._format_op(op)}")
                    print(f"  Args: {self._format_args(args)}")
                    print(f"  Fields: {n.fields}")
                    
                case Aggregate(op, init, arg, idxs):
                    self.operation_count["Aggregate"] += 1
                    print(f"\nAggregate #{self.operation_count['Aggregate']}:")
                    print(f"  Operation: {self._format_op(op)}")
                    print(f"  Init: {self._format_literal(init)}")
                    print(f"  Reduce dims: {idxs}")
                    print(f"  Input fields: {arg.fields if hasattr(arg, 'fields') else 'N/A'}")
                    print(f"  Output fields: {n.fields}")
    
    def _format_op(self, op):
        """Format operation for printing"""
        if isinstance(op, Literal):
            if hasattr(op.val, '__name__'):
                return op.val.__name__
            return str(op.val)
        return str(op)
    
    def _format_literal(self, lit):
        """Format literal for printing"""
        if isinstance(lit, Literal):
            return str(lit.val)
        return str(lit)
    
    def _format_args(self, args):
        """Format arguments for printing"""
        formatted = []
        for arg in args:
            if isinstance(arg, Alias):
                formatted.append(f"Alias({arg.name})")
            elif isinstance(arg, Table):
                formatted.append(f"Table(...)")
            else:
                formatted.append(type(arg).__name__)
        return formatted