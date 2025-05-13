from ..finch_logic import (
    Aggregate,
    Alias,
    MapJoin,
    Plan,
    Produces,
    Query,
    Subquery,
)
from ..symbolic import Chain, PostOrderDFS, PostWalk, PreWalk, Rewrite, Term
from .compiler import LogicCompiler


def optimize(prgm: Term) -> Term:
    # ...
    prgm = lift_subqueries(prgm)
    return propagate_map_queries(prgm)


def _lift_subqueries_expr(node: Term, bindings: dict[Term, Term]) -> Term:
    match node:
        case Subquery(lhs, arg):
            if lhs not in bindings:
                arg_2 = _lift_subqueries_expr(arg, bindings)
                bindings[lhs] = arg_2
            return lhs
        case any if any.is_expr():
            return any.make_term(
                any.head(),
                [_lift_subqueries_expr(x, bindings) for x in any.children()],
            )
        case _:
            return node


def lift_subqueries(node: Term) -> Term:
    match node:
        case Plan(bodies):
            return Plan(tuple(map(lift_subqueries, bodies)))
        case Query(lhs, rhs):
            bindings: dict[Term, Term] = {}
            rhs_2 = _lift_subqueries_expr(rhs, bindings)
            return Plan(
                (*[Query(lhs, rhs) for lhs, rhs in bindings.items()], Query(lhs, rhs_2))
            )
        case Produces() as p:
            return p
        case _:
            raise Exception(f"Invalid node: {node}")


def _get_productions(root: Term) -> list[Term]:
    for node in PostOrderDFS(root):
        if isinstance(node, Produces):
            return [arg for arg in PostOrderDFS(node) if isinstance(arg, Alias)]
    return []


def propagate_map_queries(root: Term) -> Term:
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


class DefaultLogicOptimizer:
    def __init__(self, ctx: LogicCompiler):
        self.ctx = ctx

    def __call__(self, prgm: Term):
        prgm = optimize(prgm)
        return self.ctx(prgm)
