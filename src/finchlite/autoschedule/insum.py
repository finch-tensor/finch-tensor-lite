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

class InsumLowerer:
    def __init__(self):
        self.el = EinsumLowerer()

    def can_optimize(self, en: ein.EinsumNode, sparse: set[str]) -> tuple[bool, dict[str, tuple[ein.Index, ...]]]:     
        """
        Checks if an einsum node can be optimized via indirect einsums.
        Specifically it checks whether node is an einsum that references any sparse tensor binding/parameter.
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
        pass

    def get_sparse_params(self, bindings: dict[str, Any]) -> set[str]:
        pass

    def optimize_plan(self, plan: ein.Plan, bindings: dict[str, Any]) -> tuple[ein.Plan, dict[str, Any]]:
        pass