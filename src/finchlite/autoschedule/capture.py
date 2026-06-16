from __future__ import annotations

from finchlite.algebra.tensor import TensorFType
from finchlite.finch_logic import (
    Alias,
    LogicLoader,
    LogicStatement,
    MockLogicLoader,
    StatsFactory,
    TensorStats,
)
from finchlite.symbolic import UnvalidatedForm


class LogicCapture(UnvalidatedForm, LogicLoader):
    def __init__(self, ctx: LogicLoader | None = None):
        if ctx is None:
            ctx = MockLogicLoader()
        self.ctx = ctx
        self.last_prgm: LogicStatement
        self.last_bindings: dict[Alias, TensorFType]
        self.last_stats: dict[Alias, TensorStats]
        self.last_stats_factory: StatsFactory

    def lower(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        self.last_prgm = prgm
        self.last_bindings = bindings.copy()
        self.last_stats = stats.copy()
        self.last_stats_factory = stats_factory
        return self.ctx(prgm, bindings, stats, stats_factory)
