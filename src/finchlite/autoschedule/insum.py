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
    init_value
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

    def optimize_einsum(self, einsum: ein.Einsum, sparse: str, sparse_idxs: tuple[ein.Index, ...]) -> list[ein.EinsumNode]:
        #bodies: list[ein.EinsumNode] = []

        # initialize mask tensor T which is a boolean that represents whether each reduced fiber in the sparse tensor has non-zero elements or not
        # Essentially T[idxs...] = whether the sparse tensor fiber being reduced at idxs... has any non-zero elements in it
        #T_idxs = tuple(idx for idx in einsum.idxs if idx in sparse_idxs)
        #bodies.append(ein.Einsum( #initialize every element of T to 0
        #    op=ein.Literal(overwrite),
        #    alias=ein.Alias(gensym(f"{sparse}_T")),
        #    idxs=T_idxs,
        #    arg=ein.Literal(0)
        #))
        #bodies.append(ein.Einsum(
        #    op=ein.Literal(operator.add),
        #    alias=ein.Alias(gensym(f"{sparse}_T")),
        #    idxs=
        #))
        pass

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