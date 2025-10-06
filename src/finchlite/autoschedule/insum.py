from multiprocessing import Value
from finchlite.autoschedule.einsum import (
    EinsumPlan,
    EinsumLowerer, 
    Einsum, 
    PointwiseAccess, 
    PointwiseNamedField, 
    PointwiseOp, 
    PointwiseLiteral, 
    GetSparseCoordArray, 
    GetSparseValueArray
)
from finchlite.autoschedule.sparse_tensor import (
    SparseTensorFType
)
from finchlite.autoschedule.einsum import EinsumLowerer
from finchlite.finch_logic import Plan, Table, Literal
from finchlite.symbolic.ftype import ftype
from finchlite.symbolic import PostWalk

class InsumLowerer:
    def __init__(self):
        self.el = EinsumLowerer()

    def can_optimize(self, einsum: Einsum, sparse_params: set[str]) -> tuple[bool, set[str]]:
        """Check if einsum accesses any sparse tensors."""
        used_sparse_params = set()
        
        def check_for_sparse_access(node):
            nonlocal used_sparse_params
            match node:
                case PointwiseAccess(PointwiseNamedField(name), _):
                    if name in sparse_params:
                        used_sparse_params.add(name)
            return None
        
        # Walk the pointwise expression tree to check for sparse accesses
        PostWalk(check_for_sparse_access)(einsum.pointwise_expr)
        
        return len(used_sparse_params) > 0, used_sparse_params

    def optimize_sparse_einsum(self, einsum: Einsum, sparse_param: str) -> Einsum:
        return einsum

    def get_sparse_params(self, parameters: dict[str, Table]) -> set[str]:
        sparse_params = set()
        
        for alias, value in parameters.items():
            match value:
                case Table(Literal(val), _):
                    if isinstance(ftype(val), SparseTensorFType):
                        sparse_params.add(alias)
                
        return sparse_params

    def __call__(self, prgm: Plan) -> tuple[EinsumPlan, dict[str, Table]]:
        einsum_plan, parameters = self.el(prgm)

        sparse_params = self.get_sparse_params(parameters)

        new_bodies = []
        for einsum in einsum_plan.bodies:
            can_optimize, used_sparse_params = self.can_optimize(einsum, sparse_params)
            if can_optimize:
                new_bodies.append(self.optimize_sparse_einsum(einsum, used_sparse_params[0]))
            else:
                new_bodies.append(einsum)

        return EinsumPlan(tuple(new_bodies), einsum_plan.returnValues), parameters