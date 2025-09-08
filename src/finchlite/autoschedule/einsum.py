from finchlite.finch_logic import LogicTree, LogicExpression, MapJoin, Aggregate, LogicNode, Alias, Table, Literal
from finchlite.autoschedule import optimize, DefaultLogicOptimizer, LogicCompiler
from finchlite.symbolic import PostOrderDFS

class Einsum(LogicTree, LogicExpression):
    pass

class Einprod(LogicTree, LogicExpression):
    pass

class EinsumTransformer(DefaultLogicOptimizer):
    """Transforms program into Einsum and Einprod"""

    def __init__(self, ctx: LogicCompiler, verbose=True):
        super().__init__(ctx)
        self.verbose = verbose
    
    def __call__(self, prgm: LogicNode):
        # First optimize the program
        prgm = optimize(prgm)

        return self.ctx(prgm)

    def transform(self, prgm: LogicNode):
        pass


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