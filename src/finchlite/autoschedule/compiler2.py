from __future__ import annotations

import operator
from typing import Any, Iterable

from finchlite.finch_notation.stages import NotationLoader
from finchlite.symbolic import gensym

from .. import finch_logic as lgc
from .. import finch_notation as ntn
from ..compile import ExtentFType
from ..finch_assembly import AssemblyLibrary
from ..finch_logic import (
    Alias,
    Field,
    Literal,
    LogicLoader,
    Relabel,
    TableValueFType,
)
from ..symbolic import Context
from .stages import LogicNotationLowerer

class PointwiseContext:
    def __init__(self, ctx: NotationContext):
        self.ctx = ctx

    def __call__(
        self,
        ex: lgc.LogicExpression,
        loops: dict[Field, ntn.Variable],
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
                        tuple(loops.values()),
                    )
                )
            case Relabel(arg, idxs):
                return self(arg, dict(zip(idxs, loops.values(), strict=True)))
            case _:
                raise Exception(f"Unrecognized logic: {ex}")

def merge_shapes(
    a: ntn.Variable | None,
    b: ntn.Variable | None
) -> ntn.Variable | None:
    if a and b:
        if a.name < b.name:
            return a
        return b
    return a or b
class NotationContext(Context):
    """
    Compiles Finch Logic to Finch Notation. Holds the state of the
    compilation process.
    """

    def __init__(
        self,
        bindings: dict[lgc.Alias, lgc.TableValueFType],
        slots: dict[lgc.Alias, ntn.Slot],
        shapes: dict[lgc.Alias, ntn.Variable],
        fields: dict[lgc.Alias, tuple[lgc.Field, ...]] | None = None,
        shape_types: dict[lgc.Alias, tuple[Any, ...]] | None = None,
        epilogue: Iterable[ntn.NotationStatement] | None = None,
    ):
        self.bindings = bindings
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
                lhs, lgc.Reorder(lgc.Relabel(Alias(_) as arg, idxs_1), idxs_2)
            ):
                # TODO (mtsokol): mostly the same as `agg`, used for explicit transpose
                raise NotImplementedError
            case lgc.Query(
                lhs, lgc.Aggregate(op, init, lgc.Reorder(arg, idxs_1), idxs_2)
            ):
                arg_shapes:tuple[ntn.Variable | None] = arg.mapdims(merge_shapes, self.shapes, self.fields)
                shapes = {idx: arg_shapes.get(idx) or ntn.Literal(1) for idx in idxs_1}
                shape_type = arg.shape_type(self.shape_types, self.fields)
                fields = arg.fields(self.fields)
                loops = {
                    idx: ntn.Variable(gensym(idx.name), t)
                    for idx, t in zip(shape_type, fields, strict=True)
                }
                ctx = PointwiseContext(self)
                rhs = ctx(arg, loops)
                lhs = ntn.Unwrap(
                    ntn.Access(
                        self.ctx.slots[lhs],
                        ntn.Update(ntn.Literal(op)),
                        tuple(loops[idx] for idx in idxs_1 if idx not in idxs_2),
                    )
                )
                body: ntn.Increment | ntn.Loop = ntn.Increment(lhs, rhs)
                for idx, t in reversed(zip(idxs_1, shape_type, strict=True)):
                    ext = (
                        ExtentFType.stack(
                            ntn.Literal(t(0)),
                            shapes[idx],
                        ),
                    )
                    body = ntn.Loop(
                        loops[idx],
                        ext,
                        body,
                    )

                return ntn.Block(
                    ntn.Declare(
                        self.ctx.slots[lhs],
                        ntn.Literal(init.val),
                        ntn.Literal(op.val),
                    ),
                    body,
                    ntn.Freeze(
                        self.ctx.slots[lhs],
                        ntn.Literal(op.val),
                    ),
                )
            case lgc.Query(lhs, lgc.Reorder(arg, idxs)):
                return self(
                    lgc.Query(
                        lhs,
                        lgc.Aggregate(
                            Literal(operator.overwrite),
                            Literal(self.bindings[lhs].tns.fill_value),
                            lgc.Reorder(arg, idxs),
                            (),
                        ),
                    )
                )
            case lgc.Produces(args):
                for arg in args:
                    assert isinstance(arg, lgc.Alias)
                return ntn.Block(
                    (
                        *self.epilogue,
                        ntn.Return(
                            ntn.Call(
                                ntn.Literal(tuple),
                                tuple(
                                    ntn.Variable(
                                        self.freshen(a.name), self.bindings[a].tns
                                    )
                                    for a in args
                                ),
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
        preamble = []
        epilogue = []
        args = {}
        slots = {}
        shapes = {}
        for arg in bindings:
            args[arg] = ntn.Variable(gensym(f"{arg.name}"), bindings[arg].tns)
            slots[arg] = ntn.Slot(gensym(f"_{arg.name}"), bindings[arg].tns)
            preamble.append(
                ntn.Unpack(
                    slots[arg],
                    args[arg],
                )
            )
            shape = []
            for i, t in enumerate(bindings[arg].tns.shape_type):
                dim = ntn.Variable(gensym(f"{arg.name}_dim_{i}"), t)
                shape.append(dim)
                preamble.append(ntn.Assign(dim, ntn.Length(slots[arg], ntn.Literal(i))))
            shapes[arg] = shape
            epilogue.append(
                ntn.Repack(
                    slots[arg],
                    args[arg],
                )
            )
        ctx = NotationContext(
            bindings,
            slots,
            shapes,
            epilogue=epilogue,
        )
        body = ctx(term)
        return ntn.Module(
            ntn.Function(
                ntn.Variable("main"),
                tuple(args.values()),
                ntn.Block(
                    (
                        *preamble,
                        body,
                    )
                ),
            )
        )


class LogicCompiler2(LogicLoader):
    def __init__(self, ctx_lower: LogicNotationLowerer, ctx_load: NotationLoader):
        self.ctx_lower = ctx_lower
        self.ctx_load = ctx_load

    def __call__(
        self, prgm: lgc.LogicStatement, bindings: dict[lgc.Alias, lgc.TableValueFType]
    ) -> tuple[AssemblyLibrary, dict[lgc.Alias, lgc.TableValueFType]]:
        mod = self.ctx_lower(prgm, bindings)
        lib = self.ctx_load(mod)
        return lib, bindings
