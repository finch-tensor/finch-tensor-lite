from dataclasses import dataclass
import operator

from finchlite.finch_logic import LogicTree, LogicExpression, MapJoin, Aggregate, LogicNode, Alias, Table, Literal, Field
from finchlite.autoschedule import DefaultLogicOptimizer, LogicCompiler, optimize
from finchlite.symbolic import Rewrite, PostWalk, PostOrderDFS

@dataclass(eq=True, frozen=False)
class Einsum(LogicTree, LogicExpression):
    """
    NumPy-style einsum logic node.

    - inputs: per-argument axis labels as Fields, e.g., ((i,k), (k,j))
    - output: output axis labels as Fields, e.g., (i,j)
    - args:   input expressions
    """

    inputs: tuple[tuple[Field, ...], ...]
    outputs: tuple[Field, ...]
    args: tuple[LogicExpression, ...]

    def __init__(self, args: tuple[LogicExpression, ...], inputs: tuple[tuple[Field, ...], ...] | None = None, outputs: tuple[Field, ...] | None = None):
        self.args = args

        #inputs are the fields of the arguments by default
        self.inputs = inputs if inputs is not None else tuple(tuple(f for f in arg.fields) for arg in args)

        #outputs are the union of the inputs by default, or the union of the outputs if provided
        union_fields: list[Field] = outputs if outputs is not None else inputs #union fields are inputs by default
        for labels in self.inputs:
            for f in labels:
                if f not in union_fields:
                    union_fields.append(f)
        self.outputs = tuple(union_fields) #outputs are simply the union of the union fields

    @property
    def children(self):
        # Treat only args as children in the term tree
        return list(self.args)

    @property
    def fields(self) -> list[Field]:
        return list(self.outputs)

    def to_string(self) -> str:
        return f"np.einsum(\"{','.join(self.inputs)}->{','.join(self.outputs)}\", {','.join(self.args)})"

class EinsumTransformer(DefaultLogicOptimizer):
    """
    Rewrite unoptimized Logic IR (mostly MapJoin and Aggregate) into Einsum nodes.

    Pattern handled:
    - Aggregate(add, 0, MapJoin(mul, args), reduce_idxs)  -> Einsum(inputs, output, args)
    - MapJoin(mul, args)                                  -> Einsum(inputs, output, args)

    After rewriting, Einsum nodes are lowered back to MapJoin/Aggregate for compilation.
    """

    def __init__(self, ctx: LogicCompiler, verbose=True):
        super().__init__(ctx)
        self.verbose = verbose
    
    def __call__(self, prgm: LogicNode):
        prgm = optimize(prgm)
        transformed = self.transform(prgm)

        return transformed

    def transform(self, prgm: LogicNode):
        def rule(node):
            match node:
                # Sum over product -> Einsum
                case Aggregate(Literal(op_add), Literal(init), MapJoin(Literal(op_mul), args), idxs):
                    if op_add is operator.add and init == 0:
                        return Einsum(args=args, inputs=None, outputs=idxs)

                # Pure elementwise product -> Einsum (no contraction)
                case MapJoin(Literal(op_mul), args):
                    if op_mul is operator.mul:
                        return Einsum(args=args, inputs=None, output=None)

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