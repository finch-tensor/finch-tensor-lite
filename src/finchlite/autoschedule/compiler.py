import operator
from functools import reduce
from typing import overload

import numpy as np

from .. import finch_assembly as asm
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
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicExpression,
    LogicNode,
    LogicStatement,
    LogicTree,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reformat,
    Relabel,
    Reorder,
    Subquery,
    Table,
    Value,
)
from ..symbolic import Fixpoint, PostWalk, Rewrite, ftype
from ._utils import extend_uniqe, intersect, setdiff, with_subsequence


@overload
def compute_structure(
    node: Field, fields: dict[str, Field], aliases: dict[str, Alias]
) -> Field: ...
@overload
def compute_structure(
    node: Alias, fields: dict[str, Field], aliases: dict[str, Alias]
) -> Alias: ...
@overload
def compute_structure(
    node: Subquery, fields: dict[str, Field], aliases: dict[str, Alias]
) -> Subquery: ...
@overload
def compute_structure(
    node: Table, fields: dict[str, Field], aliases: dict[str, Alias]
) -> Table: ...
@overload
def compute_structure(
    node: LogicTree, fields: dict[str, Field], aliases: dict[str, Alias]
) -> LogicTree: ...
@overload
def compute_structure(
    node: LogicExpression, fields: dict[str, Field], aliases: dict[str, Alias]
) -> LogicExpression: ...
@overload
def compute_structure(
    node: LogicNode, fields: dict[str, Field], aliases: dict[str, Alias]
) -> LogicNode: ...
def compute_structure(node, fields, aliases):
    match node:
        case Field(name):
            return fields.setdefault(name, Field(f"{len(fields) + len(aliases)}"))
        case Alias(name):
            return aliases.setdefault(name, Alias(f"{len(fields) + len(aliases)}"))
        case Subquery(Alias(name) as lhs, arg):
            if name in aliases:
                return aliases[name]
            arg_2 = compute_structure(arg, fields, aliases)
            lhs_2 = compute_structure(lhs, fields, aliases)
            return Subquery(lhs_2, arg_2)
        case Table(tns, idxs):
            assert isinstance(tns, Literal), "tns must be an Literal"
            return Table(
                Literal(type(tns.val)),
                tuple(compute_structure(idx, fields, aliases) for idx in idxs),
            )
        case LogicTree() as tree:
            return tree.make_term(
                tree.head(),
                *(compute_structure(arg, fields, aliases) for arg in tree.children),
            )
        case _:
            return node


class PointwiseLowerer:
    def __init__(
        self,
        bound_idxs: list[Field] | None = None,
        loop_idxs: list[Field] | None = None,
    ):
        self.bound_idxs = bound_idxs if bound_idxs is not None else []
        self.loop_idxs = loop_idxs if loop_idxs is not None else []
        self.required_slots: list[Alias] = []

    def __call__(
        self,
        ex: LogicNode,
        slot_vars: dict[Alias, ntn.Slot],
        field_relabels: dict[Field, Field],
        field_types: dict[Field, type],
    ) -> ntn.NotationExpression:
        match ex:
            case MapJoin(Literal(op), args):
                return ntn.Call(
                    ntn.Literal(op),
                    tuple(
                        self(arg, slot_vars, field_relabels, field_types)
                        for arg in args
                    ),
                )
            case Relabel(Alias(_) as alias, idxs_1):
                self.bound_idxs.extend(idxs_1)
                self.required_slots.append(alias)
                return ntn.Unwrap(
                    ntn.Access(
                        slot_vars[alias],
                        ntn.Read(),
                        tuple(
                            self(idx, slot_vars, field_relabels, field_types)
                            if idx in self.loop_idxs
                            else ntn.Value(
                                asm.Literal(field_types[idx](0)), field_types[idx]
                            )
                            for idx in idxs_1
                        ),
                    )
                )
            case Reorder(Value(ex, type_), _) | Value(ex, type_):
                return ntn.Value(ex, type_)
            case Reorder(arg, _):
                return self(arg, slot_vars, field_relabels, field_types)
            case Field(_) as f:
                return ntn.Variable(field_relabels.get(f, f).name, field_types[f])
            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def compile_pointwise_logic(
    ex: LogicNode,
    loop_idxs: list[Field],
    slot_vars: dict[Alias, ntn.Slot],
    field_relabels: dict[Field, Field],
    field_types: dict[Field, type],
) -> tuple[ntn.NotationExpression, list[Field], list[Alias]]:
    ctx = PointwiseLowerer(loop_idxs=loop_idxs)
    code = ctx(ex, slot_vars, field_relabels, field_types)
    return code, ctx.bound_idxs, ctx.required_slots


def compile_logic_constant(ex: LogicNode) -> ntn.NotationExpression:
    match ex:
        case Literal(val):
            return ntn.Literal(val)
        case Value(ex, type_):
            return ntn.Value(ex, type_)
        case _:
            raise Exception(f"Invalid constant: {ex}")


class LogicLowerer:
    def __init__(self, mode: str = "fast"):
        self.mode = mode

    @overload
    def __call__(
        self,
        ex: LogicStatement,
        table_vars: dict[Alias, ntn.Variable],
        slot_vars: dict[Alias, ntn.Slot],
        dim_size_vars: dict[ntn.Variable, ntn.Call],
        field_relabels: dict[Field, Field],
    ) -> ntn.NotationStatement: ...
    @overload
    def __call__(
        self,
        ex: LogicExpression,
        table_vars: dict[Alias, ntn.Variable],
        slot_vars: dict[Alias, ntn.Slot],
        dim_size_vars: dict[ntn.Variable, ntn.Call],
        field_relabels: dict[Field, Field],
    ) -> ntn.NotationExpression: ...
    @overload
    def __call__(
        self,
        ex: LogicNode,
        table_vars: dict[Alias, ntn.Variable],
        slot_vars: dict[Alias, ntn.Slot],
        dim_size_vars: dict[ntn.Variable, ntn.Call],
        field_relabels: dict[Field, Field],
    ) -> ntn.NotationNode: ...
    def __call__(
        self,
        ex,
        table_vars,
        slot_vars,
        dim_size_vars,
        field_relabels,
    ):
        match ex:
            case Query(Alias(name), Table(Literal(val) as tns, _)):
                return ntn.Assign(
                    ntn.Variable(
                        name,
                        BufferizedNDArrayFType(
                            NumpyBufferFType(val.dtype),
                            val.ndim,
                            TupleFType.from_tuple(val.shape_type),
                        ),
                    ),
                    compile_logic_constant(tns),
                )
            case Query(Alias(_), None):
                # we already removed tables
                return ntn.Block(())
            case Query(
                Alias(_) as lhs,
                Reformat(
                    tns, Reorder(Relabel(LogicExpression() as arg, idxs_1), idxs_2)
                ),
            ):
                loop_idxs = with_subsequence(intersect(idxs_1, idxs_2), idxs_2)
                arg_shape_type = find_suitable_rep(arg, table_vars).shape_type
                field_types = dict(zip(arg.fields(), arg_shape_type, strict=True))
                rhs, rhs_idxs, req_slots = compile_pointwise_logic(
                    Relabel(arg, idxs_1),
                    list(loop_idxs),
                    slot_vars,
                    field_relabels,
                    field_types,
                )
                # TODO (mtsokol): mostly the same as `agg`, used for explicit transpose
                raise NotImplementedError

            case Query(
                Alias(_) as lhs,
                Reformat(tns, Reorder(MapJoin(Literal(op), args), _) as reorder),
            ):
                assert isinstance(tns, TensorFType)
                # TODO (mtsokol): fetch fill value the right way
                fv = 0 if op in (operator.add, operator.sub) else 1
                return self(
                    Query(
                        lhs,
                        Reformat(
                            tns,
                            Aggregate(Literal(InitWrite(fv)), Literal(fv), reorder, ()),
                        ),
                    ),
                    table_vars,
                    slot_vars,
                    dim_size_vars,
                    field_relabels,
                )

            case Query(
                Alias(name) as lhs,
                Reformat(
                    tns,
                    Aggregate(
                        Literal(op),
                        Literal(init),
                        Reorder(LogicExpression() as arg, idxs_2),
                        idxs_1,
                    ),
                ),
            ):
                assert isinstance(tns, TensorFType)
                arg_shape_type = find_suitable_rep(arg, table_vars).shape_type
                field_types = dict(zip(arg.fields(), arg_shape_type, strict=True))
                rhs, rhs_idxs, req_slots = compile_pointwise_logic(
                    arg, list(idxs_2), slot_vars, field_relabels, field_types
                )
                lhs_idxs = setdiff(idxs_2, idxs_1)
                agg_var = ntn.Variable(name, tns)
                table_vars[lhs] = agg_var
                agg_slot = ntn.Slot(f"{name}_slot", tns)
                slot_vars[lhs] = agg_slot
                declaration = ntn.Declare(  # declare result tensor
                    agg_slot,
                    ntn.Literal(init),
                    ntn.Literal(op),
                    tuple(
                        ntn.Variable(
                            f"{field_relabels.get(idx, idx).name}_size",
                            ExtentFType(idx_type, idx_type),  # type: ignore[abstract]
                        )
                        for idx, idx_type in zip(lhs_idxs, tns.shape_type, strict=True)
                    ),
                )

                body: ntn.Block | ntn.Loop = ntn.Block(
                    (
                        ntn.Increment(
                            ntn.Access(
                                agg_slot,
                                ntn.Update(ntn.Literal(op)),
                                tuple(
                                    ntn.Variable(
                                        field_relabels.get(idx, idx).name, idx_type
                                    )
                                    for idx, idx_type in zip(
                                        lhs_idxs, tns.shape_type, strict=True
                                    )
                                ),
                            ),
                            rhs,
                        ),
                    )
                )
                for idx in reversed(idxs_2):
                    idx_type = field_types[idx]
                    idx_var = ntn.Variable(field_relabels.get(idx, idx).name, idx_type)
                    if idx in rhs_idxs:
                        body = ntn.Loop(
                            idx_var,
                            ntn.Variable(
                                f"{field_relabels.get(idx, idx).name}_size",
                                ExtentFType(idx_type, idx_type),  # type: ignore[abstract]
                            ),
                            body,
                        )
                    elif idx in lhs_idxs:
                        body = ntn.Loop(
                            idx_var,
                            ExtentFType.stack(
                                ntn.Literal(idx_type(1)),
                                ntn.Literal(idx_type(1)),
                            ),
                            body,
                        )

                return ntn.Block(
                    (
                        *[ntn.Assign(k, v) for k, v in dim_size_vars.items()],
                        *[ntn.Unpack(slot_vars[a], table_vars[a]) for a in req_slots],
                        ntn.Unpack(agg_slot, agg_var),
                        declaration,
                        body,
                        ntn.Freeze(agg_slot, ntn.Literal(op)),
                        *[ntn.Repack(slot_vars[a], table_vars[a]) for a in req_slots],
                        ntn.Repack(agg_slot, agg_var),
                    )
                )

            case Plan((Produces(args),)):
                assert len(args) == 1, "Only single return object is supported now"
                match args[0]:
                    case Reorder(Relabel(Alias(name), idxs_1), idxs_2) if set(
                        idxs_1
                    ) == set(idxs_2):
                        raise NotImplementedError("TODO: not supported")
                    case Reorder(Alias(name) as alias, _) | Relabel(
                        Alias(name) as alias, _
                    ):
                        tbl_var = table_vars[alias]
                    case Alias(name) as alias:
                        tbl_var = table_vars[alias]
                    case any:
                        raise Exception(f"Unrecognized logic: {any}")
                return ntn.Return(tbl_var)

            case Plan(bodies):
                func_block = ntn.Block(
                    tuple(
                        self(body, table_vars, slot_vars, dim_size_vars, field_relabels)
                        for body in bodies
                    )
                )
                last_statement = func_block.bodies[-1]
                match last_statement:
                    case ntn.Return(ntn.Variable(_, return_type)):
                        return ntn.Module(
                            (
                                ntn.Function(
                                    ntn.Variable("func", return_type),
                                    tuple(var for var in table_vars.values()),
                                    func_block,
                                ),
                            )
                        )
                    case _:
                        raise Exception(
                            "Last function's statement should be Return, "
                            f"but is: {last_statement}."
                        )

            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def record_tables(
    root: LogicNode,
) -> tuple[
    LogicNode,
    dict[Alias, ntn.Variable],
    dict[Alias, ntn.Slot],
    dict[ntn.Variable, ntn.Call],
    dict[Alias, Table],
    dict[Field, Field],
]:
    """
    Transforms plan from finchlite Logic to Finch Notation convention. Moves physical
    table out of the plan and memorizes dimension sizes as separate variables to
    be used in loops.
    """
    # alias to notation variable mapping
    table_vars: dict[Alias, ntn.Variable] = {}
    # notation variable to slot mapping
    slot_vars: dict[Alias, ntn.Slot] = {}
    # store loop extent variable
    dim_size_vars: dict[ntn.Variable, ntn.Call] = {}
    # actual tables
    tables: dict[Alias, Table] = {}
    # field relabels mapping to actual fields
    field_relabels: dict[Field, Field] = {}

    def rule_0(node):
        match node:
            case Query(Alias(name) as alias, Table(Literal(val), fields) as tbl):
                table_var = ntn.Variable(name, ftype(val))
                table_vars[alias] = table_var
                slot_var = ntn.Slot(f"{name}_slot", ftype(val))
                slot_vars[alias] = slot_var
                tables[alias] = tbl
                for idx, (field, field_type) in enumerate(
                    zip(fields, val.shape_type, strict=True)
                ):
                    assert isinstance(field, Field)
                    dim_size_var = ntn.Variable(
                        f"{field.name}_size", ExtentFType(field_type, field_type)
                    )
                    if dim_size_var not in dim_size_vars:
                        dim_size_vars[dim_size_var] = ntn.Call(
                            ntn.Literal(dimension), (table_var, ntn.Literal(idx))
                        )
                return Query(alias, None)

            case Query(Alias(name) as alias, rhs):
                suitable_rep = find_suitable_rep(rhs, table_vars)
                table_vars[alias] = ntn.Variable(name, suitable_rep)
                tables[alias] = Table(
                    Literal(TensorPlaceholder(dtype=suitable_rep.element_type)),
                    rhs.fields(),
                )

                return Query(alias, Reformat(suitable_rep, rhs))

            case Relabel(Alias(_) as alias, idxs) as relabel:
                field_relabels.update(
                    {
                        k: v
                        for k, v in zip(idxs, tables[alias].idxs, strict=True)
                        if k != v
                    }
                )
                return relabel

    processed_root = Rewrite(PostWalk(rule_0))(root)
    return processed_root, table_vars, slot_vars, dim_size_vars, tables, field_relabels


def find_suitable_rep(root, table_vars) -> TensorFType:
    match root:
        case MapJoin(Literal(op), args):
            args_suitable_reps_fields = [
                (find_suitable_rep(arg, table_vars), arg.fields()) for arg in args
            ]
            field_type_map: dict[Field, type] = {}
            for rep, fields in args_suitable_reps_fields:
                for st, f in zip(rep.shape_type, fields, strict=True):
                    if f in field_type_map and st != field_type_map[f]:
                        raise Exception(
                            f"Shape type mismatch for field {f}: "
                            f"{field_type_map[f]} vs {st}"
                        )
                    field_type_map[f] = st
            result_fields: tuple[Field, ...] = reduce(
                lambda acc, x: extend_uniqe(acc, x[1]), args_suitable_reps_fields, ()
            )

            dtype = np.dtype(
                return_type(
                    op,
                    *[rep.element_type for rep, _ in args_suitable_reps_fields],
                )
            )

            return BufferizedNDArrayFType(
                buf_t=NumpyBufferFType(dtype),
                ndim=np.intp(len(result_fields)),
                strides_t=TupleFType.from_tuple(
                    tuple(field_type_map[f] for f in result_fields)
                ),
            )
        case Aggregate(Literal(op), init, arg, idxs):
            init_suitable_rep = find_suitable_rep(init, table_vars)
            arg_suitable_rep = find_suitable_rep(arg, table_vars)
            buf_t = NumpyBufferFType(
                return_type(
                    op, init_suitable_rep.element_type, arg_suitable_rep.element_type
                )
            )
            strides_t = tuple(
                st
                for f, st in zip(arg.fields(), arg_suitable_rep.shape_type, strict=True)
                if f not in idxs
            )
            return BufferizedNDArrayFType(
                buf_t=buf_t,
                ndim=np.intp(len(strides_t)),
                strides_t=TupleFType.from_tuple(strides_t),
            )
        case LogicTree() as tree:
            for child in tree.children:
                suitable_rep = find_suitable_rep(child, table_vars)
                if suitable_rep is not None:
                    return suitable_rep
            raise Exception(f"Couldn't find a suitable rep for: {tree}")
        case Alias(_) as alias:
            return table_vars[alias].type_
        case Literal(val):
            return query_property(val, "asarray", "__attr__")
        case _:
            raise Exception(f"Unrecognized node: {root}")


def merge_blocks(root: ntn.NotationNode) -> ntn.NotationNode:
    """
    Removes empty blocks and flattens nested blocks. Such blocks
    appear after recording and moving physical tables out of the plan.
    """

    def rule_0(node):
        match node:
            case ntn.Block((ntn.Block(bodies), *tail)):
                return ntn.Block(bodies + tuple(tail))

    return Rewrite(PostWalk(Fixpoint(rule_0)))(root)


class LogicCompiler:
    def __init__(self):
        self.ll = LogicLowerer()

    def __call__(self, prgm: LogicNode) -> tuple[ntn.NotationNode, dict[Alias, Table]]:
        prgm, table_vars, slot_vars, dim_size_vars, tables, field_relabels = (
            record_tables(prgm)
        )
        lowered_prgm = self.ll(
            prgm, table_vars, slot_vars, dim_size_vars, field_relabels
        )
        return merge_blocks(lowered_prgm), tables
