from multiprocessing import Value
import operator
from finchlite.algebra.operator import ifelse
from finchlite.autoschedule.einsum import (
    EinsumPlan,
    EinsumLowerer, 
    Einsum, 
    PointwiseNode,
    PointwiseAccess,
    PointwiseIfElse, 
    PointwiseNamedField, 
    PointwiseOp, 
    PointwiseLiteral, 
    GetSparseCoordArray, 
    GetSparseValueArray,
    PointwiseGetDimOfIndex
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
from finchlite.symbolic import PostWalk, Rewrite
from finchlite.algebra import overwrite, init_value

class InsumLowerer:
    def __init__(self):
        self.el = EinsumLowerer()

    def can_optimize(self, einsum: Einsum, sparse_params: set[str]) -> tuple[bool, dict[str, tuple[PointwiseNamedField, ...]]]:
        """Check if einsum accesses any sparse tensors."""
        used_sparse_params = dict()
        
        def check_for_sparse_access(node):
            nonlocal used_sparse_params
            match node:
                case PointwiseAccess(PointwiseNamedField(name), idxs):
                    if name in sparse_params:
                        if name in used_sparse_params and used_sparse_params[name] != idxs:
                            raise ValueError(f"Sparse parameter {name} has different indices in the einsum")
                        used_sparse_params[name] = idxs
            return None
        
        # Walk the pointwise expression tree to check for sparse accesses
        PostWalk(check_for_sparse_access)(einsum.pointwise_expr)
        
        return len(used_sparse_params) > 0, used_sparse_params

    def optimize_sparse_einsum(self, einsum: Einsum, sparse_param: str, sparse_param_idxs: tuple[PointwiseNamedField, ...]) -> list[Einsum]:
        einsums = []

        # initialize tensor T which is a boolean tensor of whether an element exists at a particular location of a sparse tensor
        # The shape of T is equal to the shape of the NON-REDUCED indicies of the sparse tensor
        T_idxs = tuple(idx for idx in einsum.output_fields if idx in sparse_param_idxs)
        einsums.append(Einsum( 
            reduceOp=overwrite,
            output=PointwiseNamedField(f"{sparse_param}T"),
            output_fields= T_idxs,
            pointwise_expr = PointwiseLiteral(0)
        ))
        einsums.append(Einsum( 
            reduceOp=operator.add,
            output=PointwiseNamedField(f"{sparse_param}T"),
            output_fields= (GetSparseCoordArray(PointwiseNamedField(sparse_param), None),),
            pointwise_expr = PointwiseLiteral(1)
        ))

        # The indicies in the sparse tensor that are reduced; essentially reduced_idxs = sparse_param_idxs - T_idxs
        reduced_dims = tuple(PointwiseGetDimOfIndex(PointwiseNamedField(sparse_param), idx) for idx in sparse_param_idxs if idx not in T_idxs)
        reduced_prod = PointwiseOp(operator.mul, reduced_dims) if len(reduced_dims) > 0 else PointwiseLiteral(1)

        def rewrite_indicies(idxs: tuple[PointwiseNode, ...]) -> tuple[PointwiseNode, ...]:
            if idxs == sparse_param_idxs:
                return (GetSparseCoordArray(PointwiseNamedField(sparse_param), None),)
            
            new_idxs = []

            for idx in idxs:
                match idx:
                    case PointwiseNamedField(_) if idx in sparse_param_idxs:
                        new_idxs.append(GetSparseCoordArray(PointwiseNamedField(sparse_param), idx))
                    case _:
                        new_idxs.append(idx)
            return tuple(new_idxs)

        def rewrite_pointwise_expr(pointwise_expr: PointwiseNode) -> PointwiseNode:
            match pointwise_expr:
                case PointwiseAccess(PointwiseNamedField(name), idxs) if name == sparse_param and idxs == sparse_param_idxs:
                    return GetSparseValueArray(PointwiseNamedField(sparse_param))
                case PointwiseAccess(alias, idxs):
                    return PointwiseAccess(alias, rewrite_indicies(idxs))
                case _:
                    return pointwise_expr

        def sparse_is_zero(pointwise_expr: PointwiseNode) -> PointwiseNode:
            match pointwise_expr:
                case PointwiseAccess(PointwiseNamedField(name), _):
                    if name == sparse_param:
                        return PointwiseLiteral(0)
                    return pointwise_expr
            return pointwise_expr

        new_pointwise_expr = Rewrite(PostWalk(rewrite_pointwise_expr))(einsum.pointwise_expr)
        sparse_is_zero_pointwise_expr = Rewrite(PostWalk(sparse_is_zero))(new_pointwise_expr)

        # initialize the initial reduction values in output tensor
        op_init_value = init_value(einsum.reduceOp, einsum.output.element_type)
        einsums.append(Einsum(
            reduceOp=overwrite,
            output=einsum.output,
            output_fields= einsum.output_fields,
            pointwise_expr= PointwiseIfElse(
                condition= PointwiseOp(operator.eq, (PointwiseAccess(PointwiseNamedField(f"{sparse_param}T"), T_idxs), reduced_prod)),
                then_expr= PointwiseLiteral(op_init_value),
                else_expr= PointwiseOp(einsum.reduceOp,(
                    PointwiseLiteral(op_init_value),
                    sparse_is_zero_pointwise_expr
                ))
            )
        ))

        # finally we do the naive einsum -> insum
        einsums.append(Einsum(
            reduceOp=einsum.reduceOp,
            output=einsum.output,
            output_fields= rewrite_indicies(einsum.output_fields),
            pointwise_expr= new_pointwise_expr
        ))
        
        return einsums

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
                sparse_param, sparse_param_idxs = next(iter(used_sparse_params.items()))
                new_bodies.extend(self.optimize_sparse_einsum(einsum, sparse_param, sparse_param_idxs))
            else:
                new_bodies.append(einsum)

        return EinsumPlan(tuple(new_bodies), einsum_plan.returnValues), parameters