from typing import TypeVar, overload

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra.tensor import TensorFormat
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
from ..symbolic.rewriters import Fixpoint, PostWalk, Rewrite
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
    ) -> ntn.NotationNode:
        match ex:
            case MapJoin(Literal(op), args):
                return ntn.Call(
                    ntn.Literal(op), tuple(self(arg, slot_vars) for arg in args)
                )
            case Relabel(Alias(_) as alias, idxs_1):
                self.bound_idxs.extend(idxs_1)
                self.required_slots.append(alias)
                return ntn.Unwrap(
                    ntn.Access(
                        slot_vars[alias],
                        ntn.Read(),
                        tuple(
                            self(idx, slot_vars)
                            if idx in self.loop_idxs
                            else ntn.Value(asm.Literal(1), int)
                            for idx in idxs_1
                        ),
                    )
                )
            case Reorder(Value(ex, type_), _) | Value(ex, type_):
                return ntn.Value(ex, type_)
            case Reorder(arg, _):
                return self(arg, slot_vars)
            case Field(name):
                return ntn.Variable(name, int)
            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def compile_pointwise_logic(
    ex: LogicNode, loop_idxs: list[Field], slot_vars: dict[Alias, ntn.Slot]
) -> tuple[ntn.NotationNode, list[Field], list[Alias]]:
    ctx = PointwiseLowerer(loop_idxs=loop_idxs)
    code = ctx(ex, slot_vars)
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
    ) -> ntn.NotationNode:
        match ex:
            case Query(Alias(name), Table(tns, _)):
                return ntn.Assign(
                    ntn.Variable(name, type(tns)), compile_logic_constant(tns)
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
                    Relabel(arg, idxs_1), list(loop_idxs), slot_vars
                )
                # TODO: mostly the same as aggregate, used for explicit transpose
                raise NotImplementedError

            case Query(
                Alias(_) as lhs,
                Reformat(tns, Reorder(MapJoin(op, args), _) as reorder),
            ):
                assert isinstance(tns, TensorFormat)
                fv = tns.fill_value
                return self(
                    Query(
                        lhs,
                        Reformat(
                            tns,
                            Aggregate(initwrite(fv), Literal(fv), reorder, ()),
                        ),  # TODO: initwrite
                    ),
                    table_vars,
                    slot_vars,
                    dim_size_vars,
                )

            case Query(
                Alias(name) as lhs,
                Reformat(
                    tns,
                    Aggregate(Literal(op), Literal(init), Reorder(arg, idxs_2), idxs_1),
                ),
            ):
                rhs, rhs_idxs, req_slots = compile_pointwise_logic(
                    arg, list(idxs_2), slot_vars
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
                    tuple(ntn.Variable(f"{idx.name}_size", int) for idx in lhs_idxs),
                )

                body: ntn.Block | ntn.Loop = ntn.Block(
                    (
                        ntn.Increment(
                            ntn.Access(
                                agg_slot,
                                ntn.Update(ntn.Literal(op)),
                                tuple(ntn.Variable(idx.name, int) for idx in lhs_idxs),
                            ),
                            rhs,
                        ),
                    )
                )
                for idx in idxs_2:
                    if idx in rhs_idxs:
                        body = ntn.Loop(
                            ntn.Variable(idx.name, int),
                            # TODO (mtsokol): Use correct loop index type
                            ntn.Variable(f"{idx.name}_size", int),
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
                        self(body, table_vars, slot_vars, dim_size_vars)
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


def format_queries(ex: LogicNode) -> LogicNode:
    return _format_queries(ex, bindings={})


def _format_queries(ex: LogicNode, bindings: dict) -> LogicNode:
    # TODO: continue rep_construct & SuitableRep implementation
    def rep_construct(a):
        return a

    class SuitableRep:
        def __init__(self, bindings):
            pass

        def __call__(self, obj):
            return np.ndarray

    match ex:
        case Plan(bodies):
            return Plan(tuple(_format_queries(body, bindings) for body in bodies))
        case Query(lhs, rhs) if not isinstance(rhs, Reformat | Table):
            assert isinstance(rhs, LogicExpression)
            rep = SuitableRep(bindings)(rhs)
            bindings[lhs] = rep
            tns = rep_construct(rep)
            return Query(lhs, Reformat(tns, rhs))
        case Query(lhs, rhs) as query:
            bindings[lhs] = SuitableRep(bindings)(rhs)
            return query
        case _:
            return ex


def initwrite(*args):  # TODO: figure out the implementation
    raise NotImplementedError


# TODO: replace with appropriate Tensor class
_TensorType = np.ndarray


def record_tables(
    root: LogicNode,
) -> tuple[
    LogicNode,
    dict[Alias, ntn.Variable],
    dict[ntn.Variable, ntn.Slot],
    dict[ntn.Variable, ntn.Call],
    dict[Alias, _TensorType],
]:
    """
    Transforms plan from Finch Logic to Finch Notation convention. Moves physical
    table out of the plan and memorizes dimension sizes as separate variables to
    be used in loops.
    """
    # alias to notation variable mapping
    table_vars: dict[Alias, ntn.Variable] = {}
    # notation variable to slot mapping
    slot_vars: dict[ntn.Variable, ntn.Slot] = {}
    # store loop extent variable
    dim_size_vars: dict[ntn.Variable, ntn.Call] = {}
    # actual tables
    tables: dict[Alias, _TensorType] = {}

    def rule_0(node):
        match node:
            case Query(Alias(name) as alias, Table(Literal(val), fields)):
                table_var = ntn.Variable(name, type(val))
                table_vars[alias] = table_var
                slot_var = ntn.Slot(f"{name}_slot", type(val))
                slot_vars[alias] = slot_var
                tables[alias] = val
                for idx, field in enumerate(fields):
                    assert isinstance(field, Field)
                    # TODO (mtsokol): Use correct loop index type
                    dim_size_var = ntn.Variable(f"{field.name}_size", int)
                    if dim_size_var not in dim_size_vars:
                        dim_size_vars[dim_size_var] = ntn.Call(
                            ntn.Literal(dimension), (table_var, ntn.Literal(idx))
                        )

                return Query(alias, None)

    processed_root = Rewrite(PostWalk(rule_0))(root)
    return processed_root, table_vars, slot_vars, dim_size_vars, tables


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

    def __call__(
        self, prgm: LogicNode
    ) -> tuple[ntn.NotationNode, dict[Alias, _TensorType]]:
        prgm = format_queries(prgm)
        prgm, table_vars, slot_vars, dim_size_vars, tables = record_tables(prgm)
        lowered_prgm = self.ll(prgm, table_vars, slot_vars, dim_size_vars)

        for table_var in table_vars:
            # include return tables and intermediaries
            if table_var not in tables:
                tables[table_var] = np.zeros(
                    dtype=np.float64, shape=()
                )  # TODO: select correct dtype

        return merge_blocks(lowered_prgm), tables
