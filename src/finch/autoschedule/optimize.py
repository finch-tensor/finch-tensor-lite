from typing import Any, Iterable

from .compiler import LogicCompiler
from ..finch_logic import (
    Aggregate,
    Alias,
    LogicNode,
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Subquery,
)
from ..symbolic import Chain, PostOrderDFS, PostWalk, PreWalk, Rewrite


def optimize(prgm: LogicNode) -> LogicNode:
    # ...
    prgm = lift_subqueries(prgm)
    prgm = propagate_map_queries(prgm)
    return prgm


def _lift_subqueries_expr(node: LogicNode, bindings: dict) -> LogicNode:
    match node:
        case Subquery(lhs, arg):
            if lhs not in bindings:
                arg_2 = _lift_subqueries_expr(arg, bindings)
                bindings[lhs] = arg_2
            return lhs
        case any if any.is_expr():
            return any.make_term(
                any.head(),
                *map(lambda x: _lift_subqueries_expr(x, bindings), any.children()),
            )
        case _:
            return node


def lift_subqueries(node: LogicNode) -> LogicNode:
    match node:
        case Plan(bodies):
            return Plan(tuple(map(lift_subqueries, bodies)))
        case Query(lhs, rhs):
            bindings = {}
            rhs_2 = _lift_subqueries_expr(rhs, bindings)
            return Plan(
                (*[Query(lhs, rhs) for lhs, rhs in bindings.items()], Query(lhs, rhs_2))
            )
        case Produces() as p:
            return p
        case _:
            raise Exception(f"Invalid node: {node}")


def _get_productions(root: LogicNode) -> list[LogicNode]:
    for node in PostOrderDFS(root):
        if isinstance(node, Produces):
            return [arg for arg in PostOrderDFS(node) if isinstance(arg, Alias)]
    return []


def propagate_map_queries(root: LogicNode) -> LogicNode:
    def rule_agg_to_mapjoin(ex):
        match ex:
            case Aggregate(op, init, arg, ()):
                return MapJoin(op, (init, arg))

    root = Rewrite(PostWalk(rule_agg_to_mapjoin))(root)
    rets = _get_productions(root)
    props = {}
    for node in PostOrderDFS(root):
        match node:
            case Query(a, MapJoin(op, args)) if a not in rets:
                props[a] = MapJoin(op, args)

    def rule_0(ex):
        return props.get(ex)

    def rule_1(ex):
        match ex:
            case Query(a, _) if a in props:
                return Plan(())

    def rule_2(ex):
        match ex:
            case Plan(args) if Plan(()) in args:
                return Plan(tuple(a for a in args if a != Plan(())))

    root = Rewrite(PreWalk(Chain([rule_0, rule_1])))(root)
    return Rewrite(PostWalk(rule_2))(root)


def _propagate_fields(
    root: LogicNode, fields: dict[LogicNode, Iterable[LogicNode]]
) -> LogicNode:
    match root:
        case Plan(bodies):
            return Plan(tuple(_propagate_fields(b, fields) for b in bodies))
        case Query(lhs, rhs):
            rhs = _propagate_fields(rhs, fields)
            fields[lhs] = rhs.get_fields()
            return Query(lhs, rhs)
        case Alias() as a:
            return Relabel(a, tuple(fields[a]))
        case node if node.is_expr():
            return node.make_term(
                node.head(), *[_propagate_fields(c, fields) for c in node.children()]
            )
        case node:
            return node


def propagate_fields(root: LogicNode) -> LogicNode:
    return _propagate_fields(root, fields={})


class DefaultLogicOptimizer:
    def __init__(self, ctx: LogicCompiler):
        self.ctx = ctx

    def __call__(self, prgm: LogicNode):
        prgm = optimize(prgm)
        return self.ctx(prgm)
