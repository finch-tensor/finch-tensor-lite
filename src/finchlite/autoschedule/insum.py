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
from finchlite.finch_logic import (
    Plan, 
    Table, 
    Literal
)
from finchlite.symbolic.ftype import ftype
from finchlite.symbolic import PostWalk
from finchlite.algebra import overwrite

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

    def optimize_sparse_einsum(self, einsum: Einsum, sparse_param: str, sparse_param_idxs: tuple[PointwiseNamedField, ...]) -> list[Einsum]:
        einsums = []
        
        # initialize tensor T which is a boolean tensor of whether an element exists at a particular location of a sparse tensor
        einsums.append(Einsum(  # initialize T[i, j, ...] = 0
            reduceOp=overwrite,
            output=PointwiseNamedField(f"{sparse_param}T"),
            output_fields= sparse_param_idxs,
            pointwise_expr = PointwiseLiteral(0)
        ))
        einsums.append(Einsum(  # initialize T[SparseCoords[k]] = 1
            reduceOp=overwrite,
            output=PointwiseNamedField(f"{sparse_param}T"),
            output_fields= (GetSparseCoordArray(PointwiseNamedField(sparse_param), None),),
            pointwise_expr = PointwiseLiteral(1)
        ))
        
        return einsum

    def get_sparse_params(self, parameters: dict[str, Table]) -> set[str]:
        sparse_params = dict()
        
        for alias, value in parameters.items():
            match value:
                case Table(Literal(val), idxs):
                    if isinstance(ftype(val), SparseTensorFType):
                        sparse_params[alias] = idxs
                
        return sparse_params

    def __call__(self, prgm: Plan) -> tuple[EinsumPlan, dict[str, Table]]:
        einsum_plan, parameters = self.el(prgm)
        sparse_params_idxs = self.get_sparse_params(parameters)
        sparse_params = set(sparse_params_idxs.keys())

        new_bodies = []
        for einsum in einsum_plan.bodies:
            can_optimize, used_sparse_params = self.can_optimize(einsum, sparse_params)
            if can_optimize:
                new_bodies.extend(self.optimize_sparse_einsum(einsum, used_sparse_params[0], tuple(PointwiseNamedField(idx.name) for idx in sparse_params_idxs[used_sparse_params[0]])))
            else:
                new_bodies.append(einsum)

        return EinsumPlan(tuple(new_bodies), einsum_plan.returnValues), parameters