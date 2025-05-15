from textwrap import dedent
from typing import TypeVar

from ..finch_logic import (
    Alias,
    Deferred,
    Field,
    Immediate,
    LogicNode,
    MapJoin,
    NodeWithFields,
    Query,
    Reformat,
    Relabel,
    Reorder,
    Subquery,
    Table,
)
from ..symbolic import Term

T = TypeVar("T", bound="LogicNode")


def get_or_insert(dictionary: dict[str, T], key: str, default: T) -> T:
    val = dictionary.get(key)
    if val is not None:
        return val
    dictionary[key] = default
    return default


def get_structure(
    node: LogicNode, fields: dict[str, LogicNode], aliases: dict[str, LogicNode]
) -> LogicNode:
    match node:
        case Field(name):
            return get_or_insert(fields, name, Immediate(len(fields) + len(aliases)))
        case Alias(name):
            return get_or_insert(aliases, name, Immediate(len(fields) + len(aliases)))
        case Subquery(Alias(name) as lhs, arg):
            alias = aliases.get(name)
            if alias is not None:
                return alias
            in_lhs = get_structure(lhs, fields, aliases)
            assert isinstance(in_lhs, NodeWithFields)
            in_arg = get_structure(arg, fields, aliases)
            assert isinstance(in_arg, NodeWithFields)
            return Subquery(in_lhs, in_arg)
        case Table(tns, idxs):
            assert all(isinstance(idx, Field) for idx in idxs)
            return Table(
                Immediate(type(tns.val)),
                tuple(get_structure(idx, fields, aliases) for idx in idxs),  # type: ignore[misc]
            )
        case any if any.is_expr():
            return any.make_term(
                *[get_structure(arg, fields, aliases) for arg in any.children()]
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
                return (
                    f":({name}"
                    + ",".join([idx.name if idx in idxs_2 else 1 for idx in idxs_1])
                    + ")"
                )
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


def compile_logic_constant(ex: Term) -> str:
    match ex:
        case Immediate(val):
            return val
        case Deferred(ex, type_):
            return f":({ex}::{type_})"
        case _:
            raise Exception(f"Invalid constant: {ex}")


def intersect(x1: tuple, x2: tuple) -> tuple:
    return tuple(x for x in x1 if x in x2)


def with_subsequence(x1: tuple, x2: tuple) -> tuple:
    res = list(x2)
    indices = [idx for idx, val in enumerate(x2) if val in x1]
    for idx, i in enumerate(indices):
        res[i] = x1[idx]
    return tuple(res)


class LogicLowerer:
    def __init__(self, mode: str = "fast"):
        self.mode = mode

    def __call__(self, ex: Term) -> str:
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
                            {lhs.name} .= {tns.fill_value}
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

    def __call__(self, prgm: Term) -> str:
        # prgm = format_queries(prgm, True)  # noqa: F821
        return self.ll(prgm)
