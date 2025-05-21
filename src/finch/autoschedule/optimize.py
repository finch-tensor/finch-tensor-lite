from collections.abc import Iterable

from ..finch_logic import (
    Aggregate,
    Alias,
    Field,
    Immediate,
    LogicNode,
    MapJoin,
    Plan,
    Produces,
    Query,
    Reformat,
    Relabel,
    Reorder,
    Subquery,
    Table,
)
from ..symbolic import Chain, Fixpoint, PostOrderDFS, PostWalk, PreWalk, Rewrite, gensym
from ._utils import intersect, is_subsequence, setdiff, with_subsequence
from .compiler import LogicCompiler


def optimize(prgm: LogicNode) -> LogicNode:
    prgm = lift_subqueries(prgm)

    prgm = isolate_reformats(prgm)
    prgm = isolate_aggregates(prgm)
    prgm = isolate_tables(prgm)
    prgm = lift_subqueries(prgm)

    prgm = pretty_labels(prgm)

    return propagate_map_queries(prgm)


def isolate_aggregates(root: LogicNode) -> LogicNode:
    def rule_0(node):
        match node:
            case Aggregate() as agg:
                name = Alias(gensym("A"))
                return Subquery(name, agg)

    ret = PostWalk(rule_0)(root)
    assert ret is not None
    return ret


def isolate_reformats(root: LogicNode) -> LogicNode:
    def rule_0(node):
        match node:
            case Reformat() as ref:
                name = Alias(gensym("A"))
                return Subquery(name, ref)

    ret = PostWalk(rule_0)(root)
    assert ret is not None
    return ret


def isolate_tables(root: LogicNode) -> LogicNode:
    def rule_0(node):
        match node:
            case Table() as tbl:
                name = Alias(gensym("A"))
                return Subquery(name, tbl)

    ret = PostWalk(rule_0)(root)
    assert ret is not None
    return ret


def pretty_labels(root: LogicNode) -> LogicNode:
    fields: dict[Field, Field] = {}
    aliases: dict[Alias, Alias] = {}

    def rule_0(node):
        match node:
            case Field() as f:
                return fields.setdefault(f, Field(f":i{len(fields)}"))

    def rule_1(node):
        match node:
            case Alias() as a:
                return aliases.setdefault(a, Alias(f":A{len(aliases)}"))

    return Rewrite(PostWalk(Chain([rule_0, rule_1])))(root)


def _lift_subqueries_expr(
    node: LogicNode, bindings: dict[LogicNode, LogicNode]
) -> LogicNode:
    match node:
        case Subquery(lhs, arg):
            if lhs not in bindings:
                arg_2 = _lift_subqueries_expr(arg, bindings)
                bindings[lhs] = arg_2
            return lhs
        case any if any.is_expr():
            return any.make_term(
                any.head(),
                *(_lift_subqueries_expr(x, bindings) for x in any.children()),
            )
        case _:
            return node


def lift_subqueries(node: LogicNode) -> LogicNode:
    match node:
        case Plan(bodies):
            return Plan(tuple(map(lift_subqueries, bodies)))
        case Query(lhs, rhs):
            bindings: dict[LogicNode, LogicNode] = {}
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
    assert isinstance(root, LogicNode)
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


def push_fields(root):
    def rule_0(ex):
        # relabel(mapjoin(_, [1,2], [1,2]), [11,22]) =>
        #     mapjoin(
        #         _, relabel([1,2], [11,22]), relabel([1,2], [11,22])
        #     )
        match ex:
            case Relabel(MapJoin(op, args) as mj, idxs):
                reidx = dict(zip(mj.get_fields(), idxs, strict=True))
                return MapJoin(
                    op,
                    tuple(
                        Relabel(arg, tuple(reidx[f] for f in mj.get_fields()))
                        for arg in args
                    ),
                )

    def rule_1(ex):
        # relabel(agg(..., [1,2,3], 3), [11,22]) =>
        #     agg(..., relabel([1,2,3], [11,22,3]), 3)
        match ex:
            case Relabel(Aggregate(op, init, arg, agg_idxs), relabel_idxs):
                diff_idxs = setdiff(arg.get_fields(), agg_idxs)
                reidx_dict = dict(zip(diff_idxs, relabel_idxs, strict=True))
                relabeled_idxs = tuple(
                    reidx_dict.get(idx, idx) for idx in arg.get_fields()
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
                reidx_dict = dict(zip(idxs_1, idxs_2, strict=True))
                idxs_4 = tuple(reidx_dict.get(idx, idx) for idx in idxs_3)
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
            case Reorder(Reorder(arg, _), idxs):
                return Reorder(arg, idxs)

    def rule_7(ex):
        match ex:
            case Reorder(MapJoin(op, args), idxs):
                return Reorder(
                    MapJoin(
                        op,
                        tuple(
                            Reorder(arg, intersect(idxs, arg.get_fields()))
                            for arg in args
                        ),
                    ),
                    idxs,
                )

    def rule_8(ex):
        match ex:
            case Reorder(Aggregate(op, init, arg, idxs_1), idxs_2) if (
                not is_subsequence(intersect(arg.get_fields(), idxs_2), idxs_2)
            ):
                return Reorder(
                    Aggregate(
                        op,
                        init,
                        Reorder(arg, with_subsequence(idxs_2, arg.get_fields())),
                        idxs_1,
                    ),
                    idxs_2,
                )

    return Rewrite(PreWalk(Chain([Fixpoint(rule_6), rule_7, rule_8])))(root)


class DefaultLogicOptimizer:
    def __init__(self, ctx: LogicCompiler):
        self.ctx = ctx

    def __call__(self, prgm: LogicNode):
        prgm = optimize(prgm)
        return self.ctx(prgm)
