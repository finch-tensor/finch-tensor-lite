import operator
from typing import TypeVar, overload

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import InitWrite, TensorFType, query_property, return_type
from ..algebra.tensor import NDArrayFType
from ..compile import dimension
from ..finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicExpression,
    LogicNode,
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
from ._utils import intersect, setdiff, with_subsequence

T = TypeVar("T", bound="LogicNode")


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


def compute_structure(
    node: LogicNode, fields: dict[str, Field], aliases: dict[str, Alias]
) -> LogicNode:
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
    ) -> ntn.NotationNode:
        match ex:
            case MapJoin(Literal(op), args):
                return ntn.Call(
                    ntn.Literal(op),
                    tuple(self(arg, slot_vars, field_relabels) for arg in args),
                )
            case Relabel(Alias(_) as alias, idxs_1):
                self.bound_idxs.extend(idxs_1)
                self.required_slots.append(alias)
                return ntn.Unwrap(
                    ntn.Access(
                        slot_vars[alias],
                        ntn.Read(),
                        tuple(
                            self(idx, slot_vars, field_relabels)
                            if idx in self.loop_idxs
                            else ntn.Value(asm.Literal(0), int)
                            for idx in idxs_1
                        ),
                    )
                )
            case Reorder(Value(ex, type_), _) | Value(ex, type_):
                return ntn.Value(ex, type_)
            case Reorder(arg, _):
                return self(arg, slot_vars, field_relabels)
            case Field(_) as f:
                return ntn.Variable(field_relabels.get(f, f).name, int)
            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def compile_pointwise_logic(
    ex: LogicNode,
    loop_idxs: list[Field],
    slot_vars: dict[Alias, ntn.Slot],
    field_relabels,
) -> tuple[ntn.NotationNode, list[Field], list[Alias]]:
    ctx = PointwiseLowerer(loop_idxs=loop_idxs)
    code = ctx(ex, slot_vars, field_relabels)
    return code, ctx.bound_idxs, ctx.required_slots


def compile_logic_constant(ex: LogicNode) -> ntn.NotationNode:
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

    def __call__(
        self,
        ex: LogicNode,
        table_vars: dict[Alias, ntn.Variable],
        slot_vars: dict[Alias, ntn.Slot],
        dim_size_vars: dict[ntn.Variable, ntn.Call],
        field_relabels: dict[Field, Field],
    ) -> ntn.NotationNode:
        match ex:
            case Query(Alias(name), Table(tns, _)) if isinstance(tns, np.ndarray):
                return ntn.Assign(
                    ntn.Variable(name, NDArrayFType(tns.dtype, tns.ndim)),
                    compile_logic_constant(tns),
                )
            case Query(Alias(_), None):
                # we already removed tables
                return ntn.Block(())
            case Query(
                Alias(_) as lhs,
                Reformat(tns, Reorder(Relabel(Alias(_) as arg, idxs_1), idxs_2)),
            ):
                loop_idxs = with_subsequence(intersect(idxs_1, idxs_2), idxs_2)
                rhs, rhs_idxs, req_slots = compile_pointwise_logic(
                    Relabel(arg, idxs_1), list(loop_idxs), slot_vars, field_relabels
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
                    Aggregate(Literal(op), Literal(init), Reorder(arg, idxs_2), idxs_1),
                ),
            ):
                rhs, rhs_idxs, req_slots = compile_pointwise_logic(
                    arg, list(idxs_2), slot_vars, field_relabels
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
                        ntn.Variable(f"{field_relabels.get(idx, idx).name}_size", int)
                        for idx in lhs_idxs
                    ),
                )

                body: ntn.Block | ntn.Loop = ntn.Block(
                    (
                        ntn.Increment(
                            ntn.Access(
                                agg_slot,
                                ntn.Update(ntn.Literal(op)),
                                tuple(
                                    ntn.Variable(field_relabels.get(idx, idx).name, int)
                                    for idx in lhs_idxs
                                ),
                            ),
                            rhs,
                        ),
                    )
                )
                for idx in idxs_2:
                    if idx in rhs_idxs:
                        body = ntn.Loop(
                            ntn.Variable(field_relabels.get(idx, idx).name, int),
                            # TODO (mtsokol): Use correct loop index type
                            ntn.Variable(
                                f"{field_relabels.get(idx, idx).name}_size", int
                            ),
                            body,
                        )
                    elif idx in lhs_idxs:
                        body = ntn.Loop(
                            ntn.Literal(1),
                            ntn.Literal(1),
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
                return ntn.Module(
                    (
                        ntn.Function(
                            ntn.Variable("func", np.ndarray),
                            tuple(var for var in table_vars.values()),
                            func_block,
                        ),
                    )
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
    Transforms plan from Finch Logic to Finch Notation convention. Moves physical
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
                for idx, field in enumerate(fields):
                    assert isinstance(field, Field)
                    # TODO (mtsokol): Use correct loop index type
                    dim_size_var = ntn.Variable(f"{field.name}_size", int)
                    if dim_size_var not in dim_size_vars:
                        dim_size_vars[dim_size_var] = ntn.Call(
                            ntn.Literal(dimension), (table_var, ntn.Literal(idx))
                        )
                return Query(alias, None)

            case Query(Alias(name) as alias, rhs):
                suitable_rep = find_suitable_rep(rhs, table_vars)
                table_vars[alias] = ntn.Variable(name, suitable_rep)
                tables[alias] = Table(
                    Literal(
                        np.zeros(
                            dtype=suitable_rep.element_type,
                            shape=tuple(1 for _ in range(suitable_rep.ndim)),
                        )
                    ),
                    rhs.fields,
                )

                return Query(alias, Reformat(suitable_rep, rhs))

            case Relabel(Alias(_) as alias, idxs) as relabel:
                field_relabels.update(
                    {
                        k: v
                        for k, v in zip(idxs, tables[alias].idxs, strict=False)
                        if k != v
                    }
                )
                return relabel

    processed_root = Rewrite(PostWalk(rule_0))(root)
    return processed_root, table_vars, slot_vars, dim_size_vars, tables, field_relabels


def find_suitable_rep(root, table_vars) -> TensorFType:
    match root:
        case MapJoin(Literal(op), args):
            args_suitable_reps = [find_suitable_rep(arg, table_vars) for arg in args]
            return NDArrayFType(
                dtype=np.dtype(
                    return_type(
                        op,
                        *[rep.element_type for rep in args_suitable_reps],
                    )
                ),
                ndim=max(rep.ndim for rep in args_suitable_reps),
            )
        case Aggregate(Literal(op), init, arg, idxs):
            init_suitable_rep = find_suitable_rep(init, table_vars)
            arg_suitable_rep = find_suitable_rep(arg, table_vars)
            return NDArrayFType(
                dtype=np.dtype(
                    return_type(
                        op,
                        init_suitable_rep.element_type,
                        arg_suitable_rep.element_type,
                    )
                ),
                ndim=len(arg.fields) - len(idxs),
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
