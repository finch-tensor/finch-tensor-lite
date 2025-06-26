from typing import TypeVar, overload

import numpy as np

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
from ..finch_notation import nodes as ntn
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

    def __call__(self, ex) -> ntn.NotationNode:
        match ex:
            case MapJoin(Literal(val), args):
                return ntn.Call(ntn.Literal(val), tuple(self(arg) for arg in args))
            case Relabel(Alias(name), idxs_1):
                self.bound_idxs.extend(idxs_1)
                return ntn.Access(
                    ntn.Variable(name),
                    ntn.Read(),
                    tuple(
                        self(idx) if idx in self.loop_idxs else ntn.Value(1, int)
                        for idx in idxs_1
                    ),
                )
            case Reorder(Value(ex, type_), _) | Value(ex, type_):
                return ntn.Value(ex, type_)
            case Reorder(arg, _):
                return self(arg)
            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def compile_pointwise_logic(ex: LogicNode, loop_idxs: list[Field]) -> tuple:
    ctx = PointwiseLowerer(loop_idxs=loop_idxs)
    code = ctx(ex)
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

    def __call__(self, ex: LogicNode) -> ntn.NotationNode:
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
                    Relabel(arg, idxs_1), list(loop_idxs)
                )
                lhs_idxs = ...

                body = ntn.Block((ntn.Assign(),))

                for idx in loop_idxs:
                    if idx in rhs_idxs:
                        body = ntn.Loop(idx, ext, body)
                    elif idx in lhs_idxs:
                        body = ntn.Loop(idx, 1, body)

                const = compile_logic_constant(tns)
                var = ntn.Variable(name, const.result_format)
                return ntn.Block(
                    (
                        # ntn.Assign(var, const),
                        ntn.Declare(),
                        body,
                        ntn.Return(var),
                    )
                )

            case Query(
                Alias(name) as lhs, Reformat(tns, Reorder(MapJoin(op, args), idxs))
            ):
                z = tns.tns.result_type
                return self(
                    Query(
                        lhs,
                        Reformat(tns, Aggregate(1, 1, 1, 1)),  # TODO
                    )
                )

            case Query(
                Alias(name) as lhs,
                Reformat(tns, Aggregate(op, init, Reorder(arg, idxs_2), idxs_1)),
            ):
                (rhs, rhs_idxs) = compile_pointwise_logic(arg, list(idxs_2))
                lhs_idxs = [idx for idx in setdiff(idxs_2, idxs_1)]
                body = ntn.Call(ntn.Literal(val), tuple(self(arg) for arg in args))
                # TODO

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
                            (),
                            ntn.Block(tuple(self(body) for body in bodies)),
                        ),
                    )
                )

            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def format_queries(node: LogicNode) -> LogicNode:
    return _format_queries(node, bindings={})


def _format_queries(node: LogicNode, bindings: dict) -> LogicNode:
    match node:
        case Plan(bodies):
            return Plan(tuple(_format_queries(body, bindings) for body in bodies))
        case Query(lhs, rhs) if not isinstance(rhs, Reformat | Table):
            rep = SuitableRep(bindings)(rhs)
            bindings[lhs] = rep
            return Value(rep, type(rep))
        case Query(lhs, rhs) as query:
            bindings[lhs] = SuitableRep(bindings)(rhs)
            return query
        case _:
            return node


class LogicCompiler:
    def __init__(self):
        self.ll = LogicLowerer()

    def __call__(self, prgm: LogicNode) -> ntn.NotationNode:
        prgm = format_queries(prgm)
        return self.ll(prgm)
