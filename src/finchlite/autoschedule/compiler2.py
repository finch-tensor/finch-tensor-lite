from __future__ import annotations

import numpy as np
from collections.abc import Iterable
from typing import Any
import operator

from finchlite.finch_notation.stages import NotationLoader
from finchlite.symbolic import gensym
from finchlite.symbolic.traversal import PostOrderDFS

from .. import finch_logic as lgc
from .. import finch_notation as ntn
from ..finch_notation import NotationInterpreter
from ..algebra import overwrite, make_tuple
from ..compile import Extent
from ..finch_assembly import AssemblyLibrary
from ..finch_logic import (
    LogicLoader,
    TableValueFType,
)
from .stages import LogicNotationLowerer


class PointwiseContext:
    def __init__(self, ctx: NotationContext):
        self.ctx = ctx

    def __call__(
        self,
        ex: lgc.LogicExpression,
        loops: dict[lgc.Field, ntn.Variable],
    ) -> ntn.NotationExpression:
        match ex:
            case lgc.MapJoin(lgc.Literal(op), args):
                return ntn.Call(
                    ntn.Literal(op),
                    tuple(
                        self(arg, {idx: loops[idx] for idx in arg.fields()})
                        for arg in args
                    ),
                )
            case lgc.Alias(_) as var:
                return ntn.Unwrap(
                    ntn.Access(
                        self.ctx.slots[var],
                        ntn.Read(),
                        tuple(loops[idx] for idx in var.fields(self.ctx.fields)),
                    )
                )
            case lgc.Relabel(arg, idxs):
                return self(arg, {idx_1: loops[idx_2] for idx_1, idx_2 in zip(arg.fields(self.ctx.fields), idxs)})
            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def merge_shapes(a: ntn.Variable | None, b: ntn.Variable | None) -> ntn.Variable | None:
    if a and b:
        if a.name < b.name:
            return a
        return b
    return a or b


class NotationContext:
    """
    Compiles Finch Logic to Finch Notation. Holds the state of the
    compilation process.
    """

    def __init__(
        self,
        bindings: dict[lgc.Alias, lgc.TableValueFType],
        args: dict[lgc.Alias, ntn.Variable],
        slots: dict[lgc.Alias, ntn.Slot],
        shapes: dict[lgc.Alias, tuple[ntn.Variable | None, ...]],
        fields: dict[lgc.Alias, tuple[lgc.Field, ...]] | None = None,
        shape_types: dict[lgc.Alias, tuple[Any, ...]] | None = None,
        epilogue: Iterable[ntn.NotationStatement] | None = None,
    ):
        self.bindings = bindings
        self.args = args
        self.slots = slots
        self.shapes = shapes
        self.equiv: dict[ntn.Variable, ntn.Variable] = {}
        if fields is None:
            fields = {var: val.idxs for var, val in bindings.items()}
        self.fields = fields
        if shape_types is None:
            shape_types = {var: val.tns.shape_type for var, val in bindings.items()}
        self.shape_types = shape_types
        if epilogue is None:
            epilogue = ()
        self.epilogue = epilogue

    def __call__(self, prgm: lgc.LogicStatement) -> ntn.NotationStatement:
        """
        Lower Finch Notation to Finch Assembly. First we check for early
        simplifications, then we call the normal lowering for the outermost
        node.
        """
        match prgm:
            case lgc.Plan(bodies):
                return ntn.Block(tuple(self(body) for body in bodies))
            case lgc.Query(
                lhs, lgc.Reorder(lgc.Alias(_) as arg, idxs)
            ):
                return self(lgc.Query(lhs, lgc.Reorder(lgc.Relabel(arg, arg.fields(self.fields)), idxs)))
            case lgc.Query(
                lhs, lgc.Reorder(lgc.Relabel(lgc.Alias(_), idxs_1) as arg, idxs_2)
            ):
                arg_dims = arg.dimmap(merge_shapes, self.shapes, self.fields)
                shapes_map = dict(zip(idxs_1, arg_dims, strict=True))
                shapes = {idx: shapes_map.get(idx) or ntn.Literal(1) for idx in idxs_1 + idxs_2}
                arg_types = arg.shape_type(self.shape_types, self.fields)
                shape_type_map = dict(zip(idxs_1, arg_types, strict=True))
                shape_type = {idx: shape_type_map.get(idx) or np.intp for idx in idxs_1 + idxs_2}
                loop_idxs = []
                remap_idxs = {}
                out_idxs = iter(idxs_2)
                out_idx = next(out_idxs, None)
                new_idxs = []
                for idx in idxs_1:
                    loop_idxs.append(idx)
                    if idx == out_idx:
                        out_idx = next(out_idxs, None)
                        new_idxs.append(idx)
                    while (out_idx in loop_idxs or out_idx not in idxs_1) and out_idx != None:
                        if out_idx in loop_idxs:
                            new_idx = lgc.Field(gensym(f"{out_idx.name}_"))
                            remap_idxs[new_idx] = out_idx
                            loop_idxs.append(new_idx)
                            new_idxs.append(new_idx)
                        else:
                            loop_idxs.append(out_idx)
                            new_idxs.append(out_idx)
                        out_idx = next(out_idxs, None)
                while (out_idx in loop_idxs or out_idx not in idxs_1) and out_idx != None:
                    if out_idx in loop_idxs:
                        new_idx = lgc.Field(gensym(f"{out_idx.name}_"))
                        remap_idxs[new_idx] = out_idx
                        loop_idxs.append(new_idx)
                        new_idxs.append(new_idx)
                    else:
                        loop_idxs.append(out_idx)
                        new_idxs.append(out_idx)
                    out_idx = next(out_idxs, None)
                loops = {
                    idx: ntn.Variable(gensym(idx.name), shape_type.get(idx) or shape_type[remap_idxs[idx]])
                    for idx in loop_idxs
                }
                ctx = PointwiseContext(self)
                rhs = ctx(arg, loops)
                lhs_access = ntn.Access(
                    self.slots[lhs],
                    ntn.Update(ntn.Literal(overwrite)),
                    tuple(loops[idx] for idx in new_idxs),
                )
                body: ntn.NotationStatement = ntn.Increment(lhs_access, rhs)
                for idx in reversed(loop_idxs):
                    t = loops[idx].type_
                    ext = ntn.Call(ntn.Literal(Extent),
                        (ntn.Literal(t(0)),
                        shapes.get(idx) or shapes[remap_idxs[idx]]),
                    )
                    if idx in remap_idxs:
                        body = ntn.If(
                            ntn.Call(ntn.Literal(operator.eq), (loops[idx], loops[remap_idxs[idx]])),
                            body,
                        )
                    body = ntn.Loop(
                        loops[idx],
                        ext,
                        body,
                    )
                
                
                return ntn.Block(
                    (
                        ntn.Declare(
                            self.slots[lhs],
                            ntn.Literal(self.bindings[lhs].tns.fill_value),
                            ntn.Literal(overwrite),
                            (),
                        ),
                        body,
                        ntn.Freeze(
                            self.slots[lhs],
                            ntn.Literal(overwrite),
                        ),
                    )
                )
            case lgc.Query(
                lhs, lgc.Aggregate(lgc.Literal(op), lgc.Literal(init), lgc.Reorder(arg, idxs_1) as arg_2, idxs_2)
            ):
                # Build a dict mapping fields to their shapes
                arg_dims = arg_2.dimmap(merge_shapes, self.shapes, self.fields)
                shapes_map = dict(zip(idxs_1, arg_dims, strict=True))
                shapes = {idx: shapes_map.get(idx) or ntn.Literal(1) for idx in idxs_1}
                arg_types = arg_2.shape_type(self.shape_types, self.fields)
                shape_type_map = dict(zip(idxs_1, arg_types, strict=True))
                shape_type = {idx: shape_type_map.get(idx) or np.intp for idx in idxs_1}
                loops = {
                    idx: ntn.Variable(gensym(idx.name), shape_type[idx])
                    for idx in idxs_1
                }
                ctx = PointwiseContext(self)
                rhs = ctx(arg, loops)
                lhs_access = ntn.Access(
                    self.slots[lhs],
                    ntn.Update(ntn.Literal(op)),
                    tuple(loops[idx] for idx in idxs_1 if idx not in idxs_2),
                )
                body = ntn.Increment(lhs_access, rhs)
                for idx in reversed(idxs_1):
                    t = loops[idx].type_
                    ext = ntn.Call(ntn.Literal(Extent),
                        (ntn.Literal(t(0)),
                        shapes[idx]),
                    )
                    body = ntn.Loop(
                        loops[idx],
                        ext,
                        body,
                    )

                return ntn.Block(
                    (
                        ntn.Declare(
                            self.slots[lhs],
                            ntn.Literal(init),
                            ntn.Literal(op),
                            (),
                        ),
                        body,
                        ntn.Freeze(
                            self.slots[lhs],
                            ntn.Literal(op),
                        ),
                    )
                )
            case lgc.Produces(args):
                vars: list[lgc.Alias] = []
                for arg in args:
                    assert isinstance(arg, lgc.Alias)
                    vars.append(arg)
                return ntn.Block(
                    (
                        *self.epilogue,
                        ntn.Return(
                            ntn.Call(
                                ntn.Literal(make_tuple),
                                tuple(self.args[var] for var in vars),
                            )
                        ),
                    )
                )
            case _:
                raise Exception(f"Unrecognized logic: {prgm}")


class NotationGenerator(LogicNotationLowerer):
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TableValueFType]
    ) -> ntn.Module:
        preamble: list[ntn.NotationStatement] = []
        epilogue: list[ntn.NotationStatement] = []
        args: dict[lgc.Alias, ntn.Variable] = {}
        slots: dict[lgc.Alias, ntn.Slot] = {}
        shapes: dict[lgc.Alias, tuple[ntn.Variable | None, ...]] = {}
        for arg in bindings:
            args[arg] = ntn.Variable(gensym(f"{arg.name}"), bindings[arg].tns)
            slots[arg] = ntn.Slot(gensym(f"_{arg.name}"), bindings[arg].tns)
            preamble.append(
                ntn.Unpack(
                    slots[arg],
                    args[arg],
                )
            )
            shape: list[ntn.Variable] = []
            for i, t in enumerate(bindings[arg].tns.shape_type):
                dim = ntn.Variable(gensym(f"{arg.name}_dim_{i}"), t)
                shape.append(dim)
                preamble.append(
                    ntn.Assign(dim, ntn.Dimension(slots[arg], ntn.Literal(i)))
                )
            shapes[arg] = tuple(shape)
            epilogue.append(
                ntn.Repack(
                    slots[arg],
                    args[arg],
                )
            )
        ctx = NotationContext(
            bindings,
            args,
            slots,
            shapes,
            epilogue=epilogue,
        )
        body = ctx(term)
        ret_t = None
        for node in PostOrderDFS(body):
            match node:
                case ntn.Return(expr):
                    ret_t = expr.result_format
        return ntn.Module(
            (
                ntn.Function(
                    ntn.Variable("main", ret_t),
                    tuple(args.values()),
                    ntn.Block((*preamble, body)),
                ),
            )
        )


class LogicCompiler2(LogicLoader):
    def __init__(
        self,
        ctx_lower: LogicNotationLowerer | None = None,
        ctx_load: NotationLoader | None = None,
    ):
        if ctx_lower is None:
            ctx_lower = NotationGenerator()
        if ctx_load is None:
            ctx_load = NotationInterpreter()
        self.ctx_lower: LogicNotationLowerer = ctx_lower
        self.ctx_load: NotationLoader = ctx_load

    def __call__(
        self, prgm: lgc.LogicStatement, bindings: dict[lgc.Alias, lgc.TableValueFType]
    ) -> tuple[
        AssemblyLibrary, lgc.LogicStatement, dict[lgc.Alias, lgc.TableValueFType]
    ]:
        mod = self.ctx_lower(prgm, bindings)
        lib = self.ctx_load(mod)
        return lib, prgm, bindings
