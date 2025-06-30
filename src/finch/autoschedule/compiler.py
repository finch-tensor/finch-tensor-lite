from typing import TypeVar, overload

import numpy as np

from .. import finch_notation as ntn
from ..algebra.tensor import TensorFormat
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
from ..symbolic.rewriters import PostWalk, Rewrite
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

    def __call__(self, ex: LogicNode, tables: dict) -> ntn.NotationNode:
        match ex:
            case MapJoin(Literal(op), args):
                return ntn.Call(
                    ntn.Literal(op), tuple(self(arg, tables) for arg in args)
                )
            case Relabel(Alias(name), idxs_1):
                self.bound_idxs.extend(idxs_1)
                return ntn.Unwrap(
                    ntn.Access(
                        tables[Alias(name)],
                        ntn.Read(),
                        tuple(
                            self(idx, tables)
                            if idx in self.loop_idxs
                            else ntn.Value(1, int)
                            for idx in idxs_1
                        ),
                    )
                )
            case Reorder(Value(ex, type_), _) | Value(ex, type_):
                return ntn.Value(ex, type_)
            case Reorder(arg, _):
                return self(arg, tables)
            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def compile_pointwise_logic(
    ex: LogicNode, loop_idxs: list[Field], tables: dict
) -> tuple[ntn.NotationNode, list[Field]]:
    ctx = PointwiseLowerer(loop_idxs=loop_idxs)
    code = ctx(ex, tables)
    return (code, ctx.bound_idxs)


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
        self, ex: LogicNode, tables: dict, dim_sizes: dict
    ) -> ntn.NotationNode:
        match ex:
            case Query(Alias(name), Table(tns, _)):
                return ntn.Assign(
                    ntn.Variable(name, type(tns)), compile_logic_constant(tns)
                )
            case Query(
                Alias(name) as lhs,
                Reformat(tns, Reorder(Relabel(Alias(_) as arg, idxs_1), idxs_2)),
            ):
                loop_idxs = with_subsequence(intersect(idxs_1, idxs_2), idxs_2)
                (rhs, rhs_idxs) = compile_pointwise_logic(
                    Relabel(arg, idxs_1), list(loop_idxs), tables
                )
                lhs_idxs = ...
                # TODO: mostly the same as aggregate
                raise NotImplementedError

            case Query(
                Alias(name) as lhs,
                Reformat(tns, Reorder(MapJoin(op, args), _) as reorder),
            ):
                assert isinstance(tns, TensorFormat)
                fv = tns.fill_value
                return self(
                    Query(
                        lhs,
                        Reformat(
                            tns, Aggregate(initwrite(fv), Literal(fv), reorder, ())
                        ),  # TODO: initwrite
                    ),
                    tables,
                    dim_sizes,
                )

            case Query(
                Alias(name) as lhs,
                Reformat(
                    tns,
                    Aggregate(Literal(op), Literal(init), Reorder(arg, idxs_2), idxs_1),
                ),
            ):
                (rhs, rhs_idxs) = compile_pointwise_logic(arg, list(idxs_2), tables)
                lhs_idxs = tuple(idx for idx in setdiff(idxs_2, idxs_1))
                agg_res = ntn.Variable(name, type(tns))
                declaration = ntn.Declare(
                    agg_res,
                    ntn.Literal(init),
                    ntn.Literal(op),
                    tuple(ntn.Variable(f"{idx}_size") for idx in lhs_idxs),
                )

                body = ntn.Block(
                    (
                        ntn.Increment(
                            ntn.Access(
                                agg_res,
                                ntn.Update(ntn.Literal(op)),
                                lhs_idxs,
                            ),
                            rhs,
                        ),
                    )
                )
                for idx in idxs_2:
                    if idx in rhs_idxs:
                        body = ntn.Loop(
                            ntn.Variable(idx.name),
                            ntn.Variable(f"{idx.name}_size"),
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
                        *[ntn.Assign(k, v) for k, v in dim_sizes.items()],
                        ntn.Assign(agg_res, declaration),
                        body,
                        ntn.Assign(agg_res, ntn.Freeze(agg_res, ntn.Literal(op))),
                    )
                )

            case Produces(args):
                assert len(args) == 1, "Only single return object is supported"
                match args[0]:
                    case Reorder(Relabel(Alias(name), idxs_1), idxs_2) if set(
                        idxs_1
                    ) == set(idxs_2):
                        raise Exception("TODO: not supported")
                    case Reorder(Alias(name) as tns, _) | Relabel(
                        Alias(name) as tns, _
                    ):
                        arg = ntn.Variable(name)
                    case Alias(name):
                        arg = ntn.Variable(name)
                    case arg:
                        raise Exception(f"Unrecognized logic: {arg}")
                return ntn.Return(arg)

            case Plan(bodies):
                return ntn.Module(
                    (
                        ntn.Function(
                            ntn.Variable("func", np.ndarray),
                            tuple(var for var in tables.values()),
                            ntn.Block(
                                tuple(self(body, tables, dim_sizes) for body in bodies)
                            ),
                        ),
                    )
                )

            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def format_queries(node: LogicNode) -> LogicNode:
    return _format_queries(node, bindings={})


def _format_queries(node: LogicNode, bindings: dict) -> LogicNode:
    # TODO: rep_construct & SuitableRep
    def rep_construct(a):
        return a

    class SuitableRep:
        def __init__(self, bindings):
            pass

        def __call__(self, obj):
            return np.ndarray

    match node:
        case Plan(bodies):
            return Plan(tuple(_format_queries(body, bindings) for body in bodies))
        case Query(lhs, rhs) if not isinstance(rhs, Reformat | Table):
            rep = SuitableRep(bindings)(rhs)
            bindings[lhs] = rep
            tns = rep_construct(rep)
            return Query(lhs, Reformat(tns, rhs))
        case Query(lhs, rhs) as query:
            bindings[lhs] = SuitableRep(bindings)(rhs)
            return query
        case _:
            return node


def record_tables(root: LogicNode) -> tuple[LogicNode, dict, dict]:
    tables: dict[Alias, ntn.Variable] = {}
    dim_sizes: dict[ntn.Variable, ntn.Call] = {}

    def rule_0(node):
        match node:
            case Query(Alias(a), Table(Literal(val), fields) as tbl):
                table_var = ntn.Variable(a, type(val))
                tables[Alias(a)] = table_var
                for idx, field in enumerate(fields):
                    assert isinstance(field, Field)
                    dim_size_var = ntn.Variable(f"{field.name}_size", int)
                    if dim_size_var not in dim_sizes:
                        dim_sizes[dim_size_var] = ntn.Call(
                            ntn.Literal(ntn.dimension), (table_var, ntn.Literal(idx))
                        )

                return Query(Alias(a), table_var)

    return Rewrite(PostWalk(rule_0))(root), tables, dim_sizes


class LogicCompiler:
    def __init__(self):
        self.ll = LogicLowerer()

    def __call__(self, prgm: LogicNode) -> ntn.NotationNode:
        prgm = format_queries(prgm)
        prgm, tables, dim_sizes = record_tables(prgm)
        return self.ll(prgm, tables, dim_sizes)
