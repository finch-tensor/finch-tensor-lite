from .compiler import LogicCompiler
from ..finch_logic import (
    Aggregate,
    Alias,
    LogicNode,
    MapJoin,
    Immediate,
    Reorder,
    Plan,
    Produces,
    Query,
    Subquery,
    Table,
    Relabel,
)
from ..symbolic import Chain, PostOrderDFS, PostWalk, PreWalk, Rewrite, Fixpoint


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


def _tuple_diff(a: tuple, b: tuple) -> tuple:
    return tuple([i for i in a if i not in b])


def push_fields(root):
    def rule_0(ex):
        # relabel(mapjoin(_, [1,2], [1,2]), [11,22]) =>
        #     mapjoin(_, relabel(reorder([1,2], _), [11,22]), relabel(reorder([1,2], _), [11,22]))
        match ex:
            case Relabel(MapJoin(op, args) as mj, idxs):
                return MapJoin(
                    op,
                    tuple(
                        [Relabel(Reorder(arg, mj.get_fields()), idxs) for arg in args]
                    ),
                )

    def rule_1(ex):
        # relabel(agg(..., [1,2,3], 3), [11,22]) => agg(..., relabel([1,2,3], [11,22,3]), 3)
        match ex:
            case Relabel(Aggregate(op, init, arg, agg_idxs), relabel_idxs):
                diff_idxs = _tuple_diff(arg.get_fields(), agg_idxs)
                reidx_dict = dict(zip(diff_idxs, relabel_idxs))
                relabeled_idxs = tuple(
                    [reidx_dict.get(idx, idx) for idx in arg.get_fields()]
                )
                return Aggregate(op, init, Relabel(arg, relabeled_idxs), agg_idxs)

    def rule_2(ex):
        match ex:
            case Relabel(Relabel(arg, _), idxs):
                return Relabel(arg, idxs)

    def rule_3(ex):
        # relabel(reorder(_, [2,1]), [11,22]) => reorder(relabel(_, [22,11]), [11,22])
        match ex:
            case Relabel(Reorder(arg, idxs_1), idxs_2):
                idxs_3 = arg.get_fields()
                reidx_dict = dict(zip(idxs_1, idxs_2))
                idxs_4 = tuple([reidx_dict.get(idx, idx) for idx in idxs_3])
                return Reorder(Relabel(arg, idxs_4), idxs_2)

    def rule_4(ex):
        match ex:
            case Relabel(Table(arg, _), idxs):
                return Table(arg, idxs)

    def rule_5(ex):
        match ex:
            case Relabel(Immediate() as arg):
                return arg

    root = Rewrite(
        PreWalk(Fixpoint(Chain([rule_0, rule_1, rule_2, rule_3, rule_4, rule_5])))
    )(root)

    def rule_6(ex):
        match ex:
            case Reorder(MapJoin(op, args), idxs):
                return MapJoin(op, tuple([Reorder(arg, idxs) for arg in args]))

    def rule_7(ex):
        match ex:
            case Reorder(Reorder(arg, _), idxs):
                return Reorder(arg, idxs)

    root = Rewrite(PreWalk(Fixpoint(Chain([rule_6, rule_7]))))(root)

    return root


class DefaultLogicOptimizer:
    def __init__(self, ctx: LogicCompiler):
        self.ctx = ctx

    def __call__(self, prgm: LogicNode):
        prgm = optimize(prgm)
        return self.ctx(prgm)
