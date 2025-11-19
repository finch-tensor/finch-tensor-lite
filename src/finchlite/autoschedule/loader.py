from typing import TypeVar

from .. import finch_assembly as asm
from .. import finch_logic as lgc
from ..compile import NotationCompiler
from ..finch_logic import (
    LogicLoader,
    LogicNode,
    LogicNotationLowerer,
)
from ..finch_notation import NotationLoader
from .compiler import NotationGenerator

T = TypeVar("T", bound="LogicNode")


class LogicCompiler(LogicLoader):
    def __init__(
        self,
        ctx_lower: LogicNotationLowerer | None = None,
        ctx_load: NotationLoader | None = None,
    ):
        if ctx_lower is None:
            ctx_lower = NotationGenerator()
        self.ctx_lower: LogicNotationLowerer = ctx_lower
        if ctx_load is None:
            ctx_load = NotationCompiler()
        self.ctx_load: NotationLoader = ctx_load

    def __call__(
        self, prgm: LogicNode, bindings: dict[lgc.Alias, lgc.TableValueFType]
    ) -> tuple[asm.AssemblyLibrary, dict[lgc.Alias, lgc.TableValueFType]]:
        ntn_module, bindings = self.ctx_lower(prgm, bindings)
        asm_library = self.ctx_load(ntn_module)
        return asm_library, bindings
