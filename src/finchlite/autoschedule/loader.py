import operator
from functools import reduce
from typing import TypeVar, overload

import numpy as np

from ..finch_notation import NotationLoader
from ..compile import NotationCompiler

from .. import finch_assembly as asm
from .. import finch_logic as lgc
from .. import finch_notation as ntn
from ..algebra import (
    InitWrite,
    TensorFType,
    TensorPlaceholder,
    query_property,
    return_type,
)
from ..codegen import NumpyBufferFType
from ..compile import BufferizedNDArrayFType, ExtentFType, dimension
from ..finch_assembly import TupleFType
from ..finch_logic import (
    LogicExpression,
    LogicLoader,
    LogicNode,
    LogicNotationLowerer,
    LogicTree,
)
from ..symbolic import Fixpoint, PostWalk, Rewrite, ftype
from ._utils import extend_uniqe, intersect, setdiff, with_subsequence
from .compiler import NotationGenerator
from ..compile import NotationCompiler

T = TypeVar("T", bound="LogicNode")


class LogicCompiler(LogicLoader):
    def __init__(self, ctx_lower: LogicNotationLowerer | None = None, ctx_load: NotationLoader | None = None):
        if ctx_lower is None:
            ctx_lower = NotationGenerator()
        self.ctx_lower: LogicNotationLowerer = ctx_lower
        if ctx_load is None:
            ctx_load = NotationCompiler()
        self.ctx_load: NotationLoader = ctx_load

    def __call__(
        self, prgm: LogicNode, bindings: dict[lgc.Alias, lgc.TableValueFType]
    ) -> asm.AssemblyLibrary:
        ntn_module, bindings = self.ctx_lower(prgm, bindings)




