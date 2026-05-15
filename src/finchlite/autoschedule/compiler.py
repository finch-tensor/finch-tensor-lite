from __future__ import annotations

import logging
from collections.abc import Iterable

from finchlite import finch_logic as lgc
from finchlite import finch_notation as ntn
from finchlite.algebra import FinchOperator, FType, ffuncs, ftypes
from finchlite.algebra.tensor import TensorFType
from finchlite.compile.lower import make_extent
from finchlite.finch_assembly import AssemblyLibrary
from finchlite.finch_logic import (
    LogicLoader,
    StatsFactory,
    TensorStats,
    compute_shape_vars,
)
from finchlite.finch_notation import NotationInterpreter
from finchlite.finch_notation.stages import NotationLoader
from finchlite.symbolic import gensym
from finchlite.symbolic.traversal import PostOrderDFS
from finchlite.util.logging import LOG_NOTATION

from .stages import LogicNotationLowerer

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_NOTATION)


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
            case lgc.Table(lgc.Alias() as var, idxs):
                return ntn.Unwrap(
                    ntn.Access(
                        self.ctx.slots[var],
                        ntn.Read(),
                        tuple(loops[idx] for idx in idxs),
                    )
                )
            case lgc.Relabel(arg, idxs):
                return self(
                    arg,
                    {
                        idx_1: loops[idx_2]
                        for idx_1, idx_2 in zip(arg.fields(), idxs, strict=True)
                    },
                )
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
        bindings: dict[lgc.Alias, TensorFType],
        args: dict[lgc.Alias, ntn.Variable],
        slots: dict[lgc.Alias, ntn.Slot],
        shapes: dict[lgc.Alias, tuple[ntn.Variable | None, ...]],
        shape_types: dict[lgc.Alias, tuple[FType | None, ...]] | None = None,
        epilogue: Iterable[ntn.NotationStatement] | None = None,
    ):
        self.bindings = bindings
        self.args = args
        self.slots = slots
        self.shapes = shapes
        self.equiv: dict[ntn.Variable, ntn.Variable] = {}
        if shape_types is None:
            shape_types = {var: val.shape_type for var, val in bindings.items()}
        self.shape_types = shape_types
        if epilogue is None:
            epilogue = ()
        self.epilogue = epilogue

    def _lower_query_of_reorder(
        self,
        query_lhs: lgc.Alias,
        op: FinchOperator,
        arg: lgc.Table,
        reorder_idxs: tuple[lgc.Field, ...],
    ):
        arg_dims = arg.dimmap(merge_shapes, self.shapes)
        shapes_map = dict(zip(arg.idxs, arg_dims, strict=True))
        shapes = {
            idx: shapes_map.get(idx) or ntn.Literal(ftypes.intp(1))
            for idx in arg.idxs + reorder_idxs
        }
        arg_types = arg.shape_type(self.shape_types)
        shape_type_map = dict(zip(arg.idxs, arg_types, strict=True))
        shape_type = {
            idx: shape_type_map.get(idx) or ftypes.intp
            for idx in arg.idxs + reorder_idxs
        }
        loop_idxs = []
        remap_idxs = {}
        out_idxs = iter(reorder_idxs)
        out_idx = next(out_idxs, None)
        new_idxs = []
        for idx in arg.idxs:
            loop_idxs.append(idx)
            if idx == out_idx:
                out_idx = next(out_idxs, None)
                new_idxs.append(idx)
            while (
                out_idx in loop_idxs or out_idx not in arg.idxs
            ) and out_idx is not None:
                if out_idx in loop_idxs:
                    new_idx = lgc.Field(gensym(f"{out_idx.name}_"))
                    remap_idxs[new_idx] = out_idx
                    loop_idxs.append(new_idx)
                    new_idxs.append(new_idx)
                else:
                    loop_idxs.append(out_idx)
                    new_idxs.append(out_idx)
                out_idx = next(out_idxs, None)
        while (out_idx in loop_idxs or out_idx not in arg.idxs) and out_idx is not None:
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
            idx: ntn.Variable(
                gensym(idx.name),
                shape_type.get(idx) or shape_type[remap_idxs[idx]],
            )
            for idx in loop_idxs
        }
        ctx = PointwiseContext(self)
        rhs = ctx(arg, loops)
        lhs_access = ntn.Access(
            self.slots[query_lhs],
            ntn.Update(ntn.Literal(op)),
            tuple(loops[idx] for idx in new_idxs),
        )
        body: ntn.NotationStatement = ntn.Increment(lhs_access, rhs)
        for idx in reversed(loop_idxs):
            stop = shapes.get(idx) or shapes[remap_idxs[idx]]
            ext = ntn.Call(
                ntn.Literal(make_extent),
                (ntn.Literal(stop.result_type(0)), stop),
            )
            if idx in remap_idxs:
                body = ntn.If(
                    ntn.Call(
                        ntn.Literal(ffuncs.eq),
                        (loops[idx], loops[remap_idxs[idx]]),
                    ),
                    body,
                )
            body = ntn.Loop(
                loops[idx],
                ext,
                body,
            )

        return body

    def _lower_query_of_aggregate(
        self,
        query_lhs: lgc.Alias,
        agg_op: FinchOperator,
        agg_arg: lgc.Reorder,
        agg_idxs: tuple[lgc.Field, ...],
    ):
        # Build a dict mapping fields to their shapes
        arg_dims = agg_arg.dimmap(merge_shapes, self.shapes)
        shapes_map = dict(zip(agg_arg.idxs, arg_dims, strict=True))
        shapes = {idx: shapes_map.get(idx) or ntn.Literal(1) for idx in agg_arg.idxs}
        arg_types = agg_arg.shape_type(self.shape_types)
        shape_type_map = dict(zip(agg_arg.idxs, arg_types, strict=True))
        shape_type = {
            idx: shape_type_map.get(idx) or ftypes.intp for idx in agg_arg.idxs
        }
        loops = {
            idx: ntn.Variable(gensym(idx.name), shape_type[idx]) for idx in agg_arg.idxs
        }
        ctx = PointwiseContext(self)
        rhs = ctx(agg_arg.arg, loops)
        lhs_access = ntn.Access(
            self.slots[query_lhs],
            ntn.Update(ntn.Literal(agg_op)),
            tuple(loops[idx] for idx in agg_arg.idxs if idx not in agg_idxs),
        )
        body: ntn.NotationStatement = ntn.Increment(lhs_access, rhs)
        for idx in reversed(agg_arg.idxs):
            ext = ntn.Call(
                ntn.Literal(make_extent),
                (ntn.Literal(shape_type[idx](0)), shapes[idx]),
            )
            body = ntn.Loop(
                loops[idx],
                ext,
                body,
            )

        return body

    def __call__(self, prgm: lgc.LogicStatement) -> ntn.NotationStatement:
        """
        Lower Finch Logic to Finch Notation. First we check for early
        simplifications, then we call the normal lowering for the outermost
        node.
        """
        match prgm:
            case lgc.Plan(bodies):
                return ntn.Block(tuple(self(body) for body in bodies))
            case lgc.Query(lhs, lgc.Reorder(lgc.Table(lgc.Alias(), _) as arg, idxs_2)):
                body = self._lower_query_of_reorder(lhs, ffuncs.overwrite, arg, idxs_2)
                return ntn.Block(
                    (
                        ntn.Declare(
                            self.slots[lhs],
                            ntn.Literal(self.bindings[lhs].fill_value),
                            ntn.Literal(ffuncs.overwrite),
                            (),
                        ),
                        body,
                        ntn.Freeze(
                            self.slots[lhs],
                            ntn.Literal(ffuncs.overwrite),
                        ),
                    )
                )
            case lgc.Query(
                lhs,
                lgc.Aggregate(
                    lgc.Literal(op),
                    lgc.Literal(init),
                    lgc.Reorder(arg, _) as arg_2,
                    idxs_2,
                ),
            ):
                body = self._lower_query_of_aggregate(lhs, op, arg_2, idxs_2)
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
            case lgc.Query(
                lhs,
                lgc.Reorder(
                    lgc.MapJoin(
                        lgc.Literal(op),
                        (
                            lgc.Reorder(lgc.Table(lhs_1), idxs_1),
                            lgc.Reorder(lgc.Table() as tbl, idxs_2),
                        ),
                    ),
                    idxs_3,
                ),
            ) if lhs_1 == lhs and idxs_1 == idxs_3:
                body = self._lower_query_of_reorder(lhs, op, tbl, idxs_2)
                return ntn.Block(
                    (
                        ntn.Thaw(
                            self.slots[lhs],
                            ntn.Literal(op),
                        ),
                        body,
                        ntn.Freeze(
                            self.slots[lhs],
                            ntn.Literal(op),
                        ),
                    )
                )
            case lgc.Query(
                lhs,
                lgc.Reorder(
                    lgc.MapJoin(
                        lgc.Literal(op),
                        (
                            lgc.Reorder(lgc.Table(lhs_1), idxs_1),
                            lgc.Aggregate(
                                lgc.Literal(op_1),
                                lgc.Literal(init),
                                lgc.Reorder() as agg_arg,
                                agg_idxs,
                            ),
                        ),
                    ),
                    idxs_2,
                ),
            ) if lhs_1 == lhs and idxs_1 == idxs_2 and op_1 in (op, ffuncs.overwrite):
                body = self._lower_query_of_aggregate(lhs, op_1, agg_arg, agg_idxs)
                return ntn.Block(
                    (
                        ntn.Thaw(
                            self.slots[lhs],
                            ntn.Literal(op),
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
                for var in args:
                    assert isinstance(var, lgc.Alias)
                    vars.append(var)
                return ntn.Block(
                    (
                        *self.epilogue,
                        ntn.Return(
                            ntn.Call(
                                ntn.Literal(ffuncs.make_tuple),
                                tuple(self.args[var] for var in vars),
                            )
                        ),
                    )
                )
            case _:
                raise Exception(f"Unrecognized logic: {prgm}")


class NotationGenerator(LogicNotationLowerer):
    def __call__(
        self, term: lgc.LogicStatement, bindings: dict[lgc.Alias, TensorFType]
    ) -> ntn.Module:
        preamble: list[ntn.NotationStatement] = []
        epilogue: list[ntn.NotationStatement] = []
        args: dict[lgc.Alias, ntn.Variable] = {}
        slots: dict[lgc.Alias, ntn.Slot] = {}
        shapes: dict[lgc.Alias, tuple[ntn.Variable | None, ...]] = {}
        for arg in bindings:
            args[arg] = ntn.Variable(gensym(f"{arg.name}"), bindings[arg])
            slots[arg] = ntn.Slot(gensym(f"_{arg.name}"), bindings[arg])
            preamble.append(
                ntn.Unpack(
                    slots[arg],
                    args[arg],
                )
            )
            shape: list[ntn.Variable] = []
            for i, t in enumerate(bindings[arg].shape_type):
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
                    ret_t = expr.result_type
        return ntn.Module(
            (
                ntn.Function(
                    ntn.Variable("main", ret_t),
                    tuple(args.values()),
                    ntn.Block((*preamble, body)),
                ),
            )
        )


class LogicCompiler(LogicLoader):
    def __init__(
        self,
        ctx_load: NotationLoader | None = None,
        ctx_lower: LogicNotationLowerer | None = None,
    ):
        if ctx_load is None:
            ctx_load = NotationInterpreter()
        if ctx_lower is None:
            ctx_lower = NotationGenerator()
        self.ctx_load: NotationLoader = ctx_load
        self.ctx_lower: LogicNotationLowerer = ctx_lower

    def __call__(
        self,
        prgm: lgc.LogicStatement,
        bindings: dict[lgc.Alias, TensorFType],
        stats: dict[lgc.Alias, TensorStats],
        stats_factory: StatsFactory,
    ) -> tuple[
        AssemblyLibrary,
        dict[lgc.Alias, TensorFType],
        dict[lgc.Alias, tuple[lgc.Field | None, ...]],
    ]:
        mod = self.ctx_lower(prgm, bindings)
        logger.debug(mod)
        lib = self.ctx_load(mod)
        shape_vars = compute_shape_vars(prgm, bindings)
        return lib, bindings, shape_vars
