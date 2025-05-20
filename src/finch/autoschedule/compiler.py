from textwrap import dedent

from ..algebra import fill_value
from ..finch_logic import (
    Alias,
    Deferred,
    Field,
    Immediate,
    LogicNode,
    MapJoin,
    Query,
    Reformat,
    Relabel,
    Reorder,
    Subquery,
    Table,
)
from ._utils import intersect, with_subsequence


def get_or_insert(
    dictionary: dict[str, LogicNode], key: str, default: LogicNode
) -> LogicNode:
    return dictionary.setdefault(key, default)


def get_structure(
    node: LogicNode, fields: dict[str, LogicNode], aliases: dict[str, LogicNode]
) -> LogicNode:
    match node:
        case Field(name):
            return get_or_insert(fields, name, Immediate(len(fields) + len(aliases)))
        case Alias(name):
            return get_or_insert(aliases, name, Immediate(len(fields) + len(aliases)))
        case Subquery(Alias(name) as lhs, arg):
            if name in aliases:
                return aliases[name]
            return Subquery(
                get_structure(lhs, fields, aliases), get_structure(arg, fields, aliases)
            )
        case Table(tns, idxs):
            return Table(
                Immediate(type(tns.val)),
                tuple(get_structure(idx, fields, aliases) for idx in idxs),
            )
        case any if any.is_expr():
            return any.make_term(
                any.head(),
                *[get_structure(arg, fields, aliases) for arg in any.children()],
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

    def __call__(self, ex: LogicNode):
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

    def __call__(self, prgm):
        # prgm = format_queries(prgm, True)  # noqa: F821
        return self.ll(prgm)
