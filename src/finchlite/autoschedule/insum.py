from ast import alias
import operator
from typing import Any, cast

from numpy import isin
import finchlite.finch_einsum as ein
import finchlite.finch_logic as logic
from finchlite.finch_logic.nodes import Alias, Table
from finchlite.symbolic import (
    ftype,
    PostWalk,
    Rewrite,
    gensym
)
from finchlite.algebra import (
    overwrite,
    init_value,
    ifelse
)
from finchlite.autoschedule import (
    EinsumLowerer
)
from finchlite.tensor import (
    SparseTensorFType
)

class InsumLowerer:
    def __init__(self):
        self.el = EinsumLowerer()

    def can_optimize(self, en: ein.EinsumNode, sparse: set[str]) -> tuple[bool, dict[str, tuple[ein.Index, ...]]]:     
        """
        Checks if an einsum node can be optimized via indirect einsums.
        Specifically it checks whether node is an einsum that references any sparse tensor binding/parameter.

        Arguments:
            en: The einsum node to check.
            sparse: The set of aliases of sparse tensor bindings/parameters.

        Returns:
            A tuple containing:
                - A boolean indicating if the einsum node can be optimized.
                - A dictionary mapping sparse binding aliases to the indices they are referenced with.
        """
        if not isinstance(en, ein.Einsum):
            return False

        einsum = cast(ein.Einsum, en)

        refed_sparse = dict()

        def sparse_detect(node: ein.EinsumExpr):
            nonlocal refed_sparse

            match node:
                case ein.Access(ein.Alias(name), idxs):
                    if name not in sparse:
                        return None

                    if name in refed_sparse and refed_sparse[name] != idxs:
                        raise ValueError(
                            f"Sparse binding {name} is being referenced "
                            "with different indicies.")
                    refed_sparse[name] = idxs
            return None

        PostWalk(sparse_detect)(einsum.arg)

    def to_insum(self, einsum: ein.Einsum, sparse: str, sparse_idxs: tuple[ein.Index, ...]) -> list[ein.EinsumNode]:
        bodies: list[ein.EinsumNode] = []
        reduced_idx = ein.Index(gensym(f"pos"))
        # initialize mask tensor T which is a boolean that represents whether each reduced fiber in the sparse tensor has non-zero elements or not
        # Essentially T[idxs...] = whether the sparse tensor fiber being reduced at idxs... has any non-zero elements in it
        T_idxs = tuple(idx for idx in einsum.idxs if idx in sparse_idxs)
        T_mask = ein.Alias(gensym(f"{sparse}_T"))
        bodies.append(ein.Einsum( #initialize every element of T to 0
            op=ein.Literal(overwrite),
            alias=T_mask,
            idxs=T_idxs,
            arg=ein.Literal(0)
        ))
        bodies.append(ein.Einsum(
            op=ein.Literal(operator.add),
            alias=T_mask,
            idxs=(
                ein.Access(
                    ein.GetAttribute(
                        obj=ein.Alias(sparse),
                        attr=ein.Literal("coords"),
                        idx=None
                    ),
                    (reduced_idx,)          
                ),
            ),
            arg=ein.Literal(1)
        ))

        # get the reduced indicies in the sparse tensor
        reduced_idxs = tuple(idx for idx in einsum.idxs if idx not in sparse_idxs)
        
        # get the size of the fiber in the sparse tensor being reduced
        reduced_fiber_size = ein.Call(ein.Literal(operator.mul), (
            ein.Literal(1), 
            *[ein.GetAttribute(
                obj=ein.Alias(sparse),
                attr=ein.Literal("shape"),
                idx=idx
            ) for idx in reduced_idxs]
        ))

        # rewrite the indicies used to iterate over the sparse tensor
        def rewrite_indicies(idxs: tuple[ein.EinsumExpr, ...]) -> tuple[ein.EinsumExpr, ...]:
            if idxs == sparse_idxs:
                return (ein.Access(
                    ein.GetAttribute(
                        obj=ein.Alias(sparse),
                        attr=ein.Literal("coords"),
                        idx=None
                    ),
                    (reduced_idx,)          
                ),)

            new_idxs = []
            for idx in idxs:
                match idx:
                    case ein.Index(_) if idx in sparse_idxs:
                        new_idxs.append(ein.Access(
                            ein.GetAttribute(
                                obj=ein.Alias(sparse),
                                attr=ein.Literal("coords"),
                                idx=idx
                            ),
                            (reduced_idx,)          
                        ))
                    case _:
                        new_idxs.append(idx)
            return tuple(new_idxs)

        # pattern matching rule to rewrite all indicies in arg
        def rewrite_all_indicies(node: ein.EinsumExpr) -> ein.EinsumExpr:
            match node:
                case ein.Access(ein.Alias(name), idxs) if name == sparse and idxs == sparse_idxs:
                    return ein.Access(
                        ein.GetAttribute(
                            obj=ein.Alias(sparse),
                            attr=ein.Literal("elems"),
                            idx=None
                        ),
                        (reduced_idx,)          
                    )
                case ein.Access(ein.Alias(name), idxs):
                    return ein.Access(ein.Alias(name), rewrite_indicies(idxs))
        
        # rewrite a pointwise expression to assume that the sparse tensor is all-zero
        def rewrite_zero(node: ein.EinsumExpr) -> ein.EinsumExpr:
            match node:
                case ein.Access(ein.Alias(name), _) if name == sparse:
                    return ein.Literal(0)
                case ein.Access(ein.GetAttribute(ein.Alias(name), ein.Literal("elems"), None), _) if name == sparse:
                    return ein.Literal(0)

        # rewrite 
        new_einarg = Rewrite(PostWalk(rewrite_all_indicies))(einsum.arg)
        zero_einarg = Rewrite(PostWalk(rewrite_zero))(einsum.arg)

        # initialize the reduction values
        # essentially, we calculate the reduction values for the reduced fibers of the sparse tensor that are non zero, and hence who's iterations asre skipped
        # we make the following core assumption: that the reduction operator, $f$ is associative and commutative. 
        # In other words, $f(a, f(b, c)) = f(f(a, b), c)$ for all $a, b, c$.
        # In essence we assume a single zero element combined with the initial value passed through the reduction operator will 
        # be equal to the effect of one or more zero elements at any point in the reduced fiber combined with the initial value.
        init = 0 if einsum.op == overwrite else init_value(einsum.op, type(0))
        bodies.append(ein.Einsum(
            op=ein.Literal(overwrite),
            alias=einsum.alias,
            idxs=einsum.idxs,
            arg=ein.Call(ein.Literal(ifelse), (
                ein.Call(ein.Literal(operator.eq), ( #check if T[idxs...] == reduced_fiber_size
                    ein.Access(T_mask, (reduced_idx,)),
                    reduced_fiber_size
                )),
                init, # if fiber is all non-zero initial reduction value is default
                ein.Call(ein.Literal(einsum.op), ( 
                    ein.Literal(init),
                    zero_einarg
                ))
            ))
        ))

        #finally we execute the naive einsum -> insum
        bodies.append(ein.Einsum(
            op=einsum.op,
            alias=einsum.alias,
            idxs=rewrite_indicies(einsum.idxs),
            arg=new_einarg
        ))

        return bodies

    def get_sparse_params(self, bindings: dict[str, Any]) -> set[str]:
        """
        Gets the set of sparse binding aliases from the bindings dictionary.

        Arguments:
            bindings: The bindings dictionary.

        Returns:
            A set of sparse binding aliases.
        """
        
        sparse = set()

        for alias, value in bindings.items():
            match value:
                case logic.Table(logic.Literal(tensor_value), _):
                    if isinstance(ftype(tensor_value), SparseTensorFType):
                        sparse.add(alias)

        return sparse

    def optimize_plan(self, plan: ein.Plan, bindings: dict[str, Any]) -> tuple[ein.Plan, dict[str, Any]]:
        pass