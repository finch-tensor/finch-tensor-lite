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
from finchlite.autoschedule.einsum import EinsumLowerer
from finchlite.finch_logic import Plan, Table

class InsumLowerer:
    def __init__(self):
        self.el = EinsumLowerer()

    def can_optimize(self, einsum: Einsum) -> bool:
        return False

    def optimize_sparse_einsum(self, einsum: Einsum) -> Einsum:
        return einsum

    def __call__(self, prgm: Plan) -> tuple[EinsumPlan, dict[str, Table]]:
        einsum_plan, parameters = self.el(prgm)

        new_bodies = []
        for einsum in einsum_plan.bodies:
            if self.can_optimize(einsum):
                new_bodies.append(self.optimize_sparse_einsum(einsum))
            else:
                new_bodies.append(einsum)

        return EinsumPlan(tuple(new_bodies), einsum_plan.returnValues), parameters