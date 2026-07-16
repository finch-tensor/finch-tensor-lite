"""
SuitableRep analyzes query expressions and predicts the representation
of results.

Corresponds to the SuitableRep struct in Finch.jl.
"""

import logging
from abc import abstractmethod
from typing import Any

import numpy as np

from finchlite import finch_logic as lgc
from finchlite.algebra import TensorFType
from finchlite.finch_assembly import AssemblyLibrary
from finchlite.finch_logic import LogicLoader, MockLogicLoader
from finchlite.finch_logic.nodes import (
    Alias,
)
from finchlite.finch_logic.tensor_stats import StatsFactory, TensorStats
from finchlite.util.logging import LOG_LOGIC_POST_OPT

from .formatter import LogicFormatter

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)

class SmartLogicFormatter(LogicFormatter):
    def __init__(self, loader: LogicLoader | None = None):
        super().__init__()
        if loader is None:
            loader = MockLogicLoader()
        self.loader = loader

    @abstractmethod
    def get_output_tns_type(
        self, fill_value: Any, shape_type: tuple[Any, ...], rep: Representation
    ): ...

    def __call__(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory[Any],
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        dict[lgc.Alias, tuple[lgc.Field | None, ...]],
    ]:
        bindings = bindings.copy()
        suitable_rep = SuitableRep(
            bindings={alias: data_rep(ftype) for alias, ftype in bindings.items()}
        )
        fill_values = prgm.infer_fill_value(
            {var: val.fill_value for var, val in bindings.items()}
        )
        shape_types = prgm.infer_shape_type(
            {var: val.shape_type for var, val in bindings.items()}
        )

        def formatter(node: lgc.LogicStatement):
            match node:
                case lgc.Plan(bodies):
                    for body in bodies:
                        formatter(body)
                case lgc.Query(lhs, rhs):
                    if lhs not in bindings:
                        rep = suitable_rep(rhs)
                        shape_type = tuple(
                            dim if dim is not None else np.intp
                            for dim in shape_types[lhs]
                        )
                        tns = self.get_output_tns_type(
                            fill_values[lhs], shape_type, rep
                        )

                        bindings[lhs] = tns
                        suitable_rep.bindings[lhs] = rep
                case lgc.Produces(_):
                    pass
                case _:
                    raise ValueError(
                        f"Unsupported logic statement for formatting: {node}"
                    )

        formatter(prgm)

        logger.debug(prgm)

        lib, bindings, shape_vars = self.loader(prgm, bindings, stats, stats_factory)
        return lib, bindings, shape_vars
