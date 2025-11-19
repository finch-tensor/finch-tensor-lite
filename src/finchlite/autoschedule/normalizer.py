from ..symbolic import gensym, Namespace
from .compiler import LogicCompiler
from ..finch_logic import LogicEvaluator, LogicLoader

import operator
from functools import reduce
from typing import TypeVar, overload

import numpy as np

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
    LogicTransform,
    TableValueFType,
    TableValue,
    Alias,
    Field,
)
from ..symbolic import Fixpoint, PostWalk, Rewrite, ftype
from ._utils import extend_uniqe, intersect, setdiff, with_subsequence

class LogicNormalizer(LogicEvaluator):
    def __init__(self, ctx: LogicEvaluator):
        self.ctx: LogicEvaluator = ctx

    def __call__(
        self, prgm: LogicNode, bindings: dict[Alias, TableValue]
    ) -> tuple[LogicNode, dict[Alias, TableValue]]:
        spc = Namespace(prgm)
        for var in bindings.keys():
            spc.freshen(var.name)
        renames = {}
        unrenames = {}
        def rule_0(node: LogicNode) -> LogicNode:
            match node:
                case Alias(name):
                    if name in renames:
                        return Alias(renames[name])
                    else:
                        new_name = spc.freshen("A")
                        renames[name] = new_name
                        return Alias(new_name)
                case Field(name):
                    if name in renames:
                        return Field(renames[name])
                    else:
                        new_name = spc.freshen("i")
                        renames[name] = new_name
                        unrenames[new_name] = name
                        return Field(new_name)
                case _:
                    return None

        root = Rewrite(PostWalk(rule_0))(prgm)

        def reidx(tbl:TableValue, names):
            return TableValue(tbl.tns, tuple(
                Field(names[idx.name]) for idx in tbl.idxs
            ))

        bindings = {rule_0(var): reidx(tbl, renames) for var, tbl in bindings.items()}
        res = self.ctx(root, bindings)

        if isinstance(res, tuple):
            return tuple(reidx(tbl, unrenames) for tbl in res)
        else:
            return reidx(res, unrenames)