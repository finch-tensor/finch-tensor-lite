import operator
from typing import Any
import finchlite.finch_einsum as ein
import finchlite.finch_logic as logic
from finchlite.finch_logic.nodes import Table
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

    def can_optimize(self, node: ein.EinsumNode, sparse_params: set[str]) -> tuple[bool, dict[str, tuple[ein.Index, ...]]]:
        pass

    def optimize_einsum(self, einsum: ein.Einsum, sparse_param: str, sparse_param_idxs: tuple[ein.Index, ...]) -> list[ein.EinsumNode]:
        pass

    def get_sparse_params(self, parameters: dict[str, Table]) -> set[str]:
        pass

    def optimize_plan(self, plan: ein.Plan, parameters: dict[str, Any]) -> tuple[ein.Plan, dict[str, Any]]:
        pass