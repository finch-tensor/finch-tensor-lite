from textwrap import dedent
from typing import TypeVar, overload

from ..algebra import fill_value
from ..finch_logic import (
    Alias,
    Deferred,
    Field,
    Immediate,
    LogicExpression,
    LogicNode,
    LogicTree,
    MapJoin,
    Query,
    Reformat,
    Relabel,
    Reorder,
    Subquery,
    Table,
)
from ._utils import intersect, with_subsequence

T = TypeVar("T", bound="LogicNode")


@overload
def get_structure(
    node: Field, fields: dict[str, Field], aliases: dict[str, Alias]
) -> Field: ...


@overload
def get_structure(
    node: Alias, fields: dict[str, Field], aliases: dict[str, Alias]
) -> Alias: ...


@overload
def get_structure(
    node: Subquery, fields: dict[str, Field], aliases: dict[str, Alias]
) -> Subquery: ...


@overload
def get_structure(
    node: Table, fields: dict[str, Field], aliases: dict[str, Alias]
) -> Table: ...


@overload
def get_structure(
    node: LogicTree, fields: dict[str, Field], aliases: dict[str, Alias]
) -> LogicTree: ...


@overload
def get_structure(
    node: LogicExpression, fields: dict[str, Field], aliases: dict[str, Alias]
) -> LogicExpression: ...


@overload
def get_structure(
    node: LogicNode, fields: dict[str, Field], aliases: dict[str, Alias]
) -> LogicNode: ...


def get_structure(
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
            arg_2 = get_structure(arg, fields, aliases)
            lhs_2 = get_structure(lhs, fields, aliases)
            return Subquery(lhs_2, arg_2)
        case Table(tns, idxs):
            assert isinstance(tns, Immediate), "tns must be an Immediate"
            return Table(
                Immediate(type(tns.val)),
                tuple(get_structure(idx, fields, aliases) for idx in idxs),
            )
        case LogicTree() as tree:
            return tree.make_term(
                tree.head(),
                *(get_structure(arg, fields, aliases) for arg in tree.children()),
            )
        case _:
            return node


class PointwiseLowerer:
    def __init__(self):
        self.bound_idxs = []

    def __call__(self, ex):
        match ex:
            case MapJoin(Immediate(val), args):
                return f":({val}({','.join([self(arg) for arg in args])}))"
            case Reorder(Relabel(Alias(name), idxs_1), idxs_2):
                self.bound_idxs.append(idxs_1)
                idxs_str = ",".join(
                    [idx.name if idx in idxs_2 else 1 for idx in idxs_1]
                )
                return f":({name}[{idxs_str}])"
            case Reorder(Immediate(val), _):
                return val
            case Immediate(val):
                return val
            case _:
                raise Exception(f"Unrecognized logic: {ex}")


def compile_pointwise_logic(ex: LogicNode) -> tuple:
    ctx = PointwiseLowerer()
    code = ctx(ex)
    return (code, ctx.bound_idxs)


def compile_logic_constant(ex: LogicNode) -> str:
    match ex:
        case Immediate(val):
            return val
        case Deferred(ex, type_):
            return f":({ex}::{type_})"
        case _:
            raise Exception(f"Invalid constant: {ex}")


class LogicLowerer:
    def __init__(self, mode: str = "fast"):
        self.mode = mode

    def __call__(self, ex: LogicNode) -> str:
        match ex:
            case Query(Alias(name), Table(tns, _)):
                return f":({name} = {compile_logic_constant(tns)})"

            case Query(
                Alias(_) as lhs,
                Reformat(tns, Reorder(Relabel(Alias(_) as arg, idxs_1), idxs_2)),
            ):
                loop_idxs = [
                    idx.name
                    for idx in with_subsequence(intersect(idxs_1, idxs_2), idxs_2)
                ]
                lhs_idxs = [idx.name for idx in idxs_2]
                (rhs, rhs_idxs) = compile_pointwise_logic(
                    Reorder(Relabel(arg, idxs_1), idxs_2)
                )
                body = f":({lhs.name}[{','.join(lhs_idxs)}] = {rhs})"
                for idx in loop_idxs:
                    if Field(idx) in rhs_idxs:
                        body = f":(for {idx} = _ \n {body} end)"
                    elif idx in lhs_idxs:
                        body = f":(for {idx} = 1:1 \n {body} end)"

                result = f"""\
                    quote
                        {lhs.name} = {compile_logic_constant(tns)}
                        @finch mode = {self.mode} begin
                            {lhs.name} .= {fill_value(tns)}
                            {body}
                            return {lhs.name}
                        end
                    end
                    """
                return dedent(result)

            # TODO: ...

            case _:
                raise Exception(f"Unrecognized logic: {ex}")


class LogicCompiler:
    def __init__(self):
        self.ll = LogicLowerer()

    def __call__(self, prgm: LogicNode) -> str:
        # prgm = format_queries(prgm, True)  # noqa: F821
        return self.ll(prgm)
