import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from typing import TypeVar, overload

from finch.algebra.algebra import is_annihilator, is_distributive, is_identity

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
)
from ..finch_logic._utils import NonConcordantLists, merge_concordant
from ..symbolic import (
    Chain,
    Fixpoint,
    Namespace,
    PostOrderDFS,
    PostWalk,
    PreWalk,
    Rewrite,
    gensym,
)
from ._utils import intersect, is_subsequence, setdiff, with_subsequence
from .compiler import LogicCompiler

T = TypeVar("T", bound="LogicNode")


def optimize(prgm: LogicNode) -> LogicNode:
    prgm = lift_subqueries(prgm)

    prgm = propagate_map_queries_backward(prgm)

    prgm = isolate_reformats(prgm)
    prgm = isolate_aggregates(prgm)
    prgm = isolate_tables(prgm)
    prgm = lift_subqueries(prgm)

    prgm = pretty_labels(prgm)

    prgm = propagate_fields(prgm)
    prgm = propagate_copy_queries(prgm)
    prgm = propagate_transpose_queries(prgm)
    prgm = propagate_map_queries(prgm)

    prgm = propagate_fields(prgm)
    prgm = push_fields(prgm)
    prgm = lift_fields(prgm)
    prgm = push_fields(prgm)

    prgm = propagate_transpose_queries(prgm)
    # prgm = set_loop_order(prgm)
    prgm = push_fields(prgm)

    prgm = concordize(prgm)

    prgm = materialize_squeeze_expand_productions(prgm)
    prgm = propagate_copy_queries(prgm)

    prgm = propagate_into_reformats(prgm)
    prgm = propagate_copy_queries(prgm)

    return normalize_names(prgm)


def isolate_aggregates(root: LogicNode) -> LogicNode:
    def rule_0(node):
        match node:
            case Aggregate() as agg:
                name = Alias(gensym("A"))
                return Subquery(name, agg)

    return Rewrite(PostWalk(rule_0))(root)


def isolate_reformats(root: LogicNode) -> LogicNode:
    def rule_0(node):
        match node:
            case Reformat() as ref:
                name = Alias(gensym("A"))
                return Subquery(name, ref)

    return Rewrite(PostWalk(rule_0))(root)


def isolate_tables(root: LogicNode) -> LogicNode:
    def rule_0(node):
        match node:
            case Table() as tbl:
                name = Alias(gensym("A"))
                return Subquery(name, tbl)

    return Rewrite(PostWalk(rule_0))(root)


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


@overload
def _lift_subqueries_expr(  # type: ignore[overload-overlap]
    node: Subquery, bindings: dict[LogicNode, LogicNode]
) -> LogicExpression: ...


@overload
def _lift_subqueries_expr(
    node: LogicTree, bindings: dict[LogicNode, LogicNode]
) -> LogicTree: ...


@overload
def _lift_subqueries_expr(
    node: LogicNode, bindings: dict[LogicNode, LogicNode]
) -> LogicNode: ...


def _lift_subqueries_expr(
    node: LogicNode, bindings: dict[LogicNode, LogicNode]
) -> LogicNode:
    match node:
        case Subquery(lhs, arg):
            if lhs not in bindings:
                arg_2 = _lift_subqueries_expr(arg, bindings)
                bindings[lhs] = arg_2
            return lhs
        case LogicTree() as tree:
            return tree.make_term(
                tree.head(),
                *(_lift_subqueries_expr(x, bindings) for x in tree.children),
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
            raise Exception(f"Invalid node: {node} in lift_subqueries")


def _collect_productions(root: LogicNode) -> list[LogicNode]:
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
    rets = _collect_productions(root)
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


def propagate_map_queries_backward(root):
    def rule_0(ex):
        match ex:
            case Aggregate(op, init, arg, ()):
                return MapJoin(op, (init, arg))

    root = Rewrite(PostWalk(rule_0))(root)

    uses: dict[LogicNode, int] = {}
    defs: dict[LogicNode, LogicNode] = {}
    rets = _collect_productions(root)
    for node in PostOrderDFS(root):
        match node:
            case Alias() as a:
                uses[a] = uses.get(a, 0) + 1
            case Query(a, b):
                uses[a] = uses.get(a, 0) - 1
                defs[a] = b

    def rule_1(ex):
        match ex:
            case Query(a, _) if uses[a] == 1 and a not in rets:
                return Plan()

    def rule_2(ex):
        match ex:
            case a if uses.get(a, 0) == 1 and a not in rets:
                return defs.get(a, a)

    root = Rewrite(PreWalk(Chain([rule_1, rule_2])))(root)
    root = push_fields(root)

    def rule_3(ex):
        match ex:
            case MapJoin(
                Literal() as f,
                args,
            ):
                for idx, item in reversed(list(enumerate(args))):
                    before_item = args[:idx]
                    after_item = args[idx + 1 :]
                    match item:
                        case (
                            Aggregate(
                                Literal() as g, Literal() as init, arg, idxs
                            ) as agg
                        ) if (
                            is_distributive(f.val, g.val)
                            and is_annihilator(f.val, init.val)
                            and len(agg.fields)
                            == len(MapJoin(f, (*before_item, *after_item)).fields)
                        ):
                            return Aggregate(
                                g,
                                init,
                                MapJoin(f, (*before_item, arg, *after_item)),
                                idxs,
                            )
                return None

    def rule_4(ex):
        match ex:
            case Aggregate(
                Literal() as op_1,
                Literal() as init_1,
                Aggregate(op_2, Literal() as init_2, arg, idxs_1),
                idxs_2,
            ) if op_1 == op_2 and is_identity(op_2.val, init_2.val):
                return Aggregate(op_1, init_1, arg, idxs_1 + idxs_2)

    def rule_5(ex):
        match ex:
            case Reorder(Aggregate(op, init, arg, idxs_1), idxs_2):
                merged_idxs: list[Field]
                try:
                    merged_idxs = merge_concordant([arg.fields, idxs_1, idxs_2])
                except NonConcordantLists:
                    merged_idxs = list(idxs_2 + idxs_1)
                return Aggregate(op, init, Reorder(arg, tuple(merged_idxs)), idxs_1)

    return Rewrite(Fixpoint(PreWalk(Chain([rule_3, rule_4, rule_5]))))(root)


def propagate_copy_queries(root):
    copies = {}
    for node in PostOrderDFS(root):
        match node:
            case Query(lhs, Alias(_) as rhs):
                copies[lhs] = copies.get(rhs, rhs)

    def rule_0(ex):
        match ex:
            case Query(lhs, rhs) if lhs == rhs:
                return Plan()

    return Rewrite(PostWalk(Chain([lambda node: copies.get(node), rule_0])))(root)


def propagate_into_reformats(root):
    @dataclass
    class Entry:
        node: Query
        node_pos: int
        matched: Query | None = None
        matched_pos: int | None = None

    def rule_0(ex):
        match ex:
            case Plan(bodies):
                queries: list[Entry] = []
                for idx, node in enumerate(bodies):
                    match node:
                        case Query(_, Reformat(_, arg)) as que_ref:
                            for q in queries[::-1]:
                                if q.node.lhs == arg:
                                    q.matched = que_ref
                                    q.matched_pos = idx
                                    break
                        case Query(_, _) as q:
                            queries.append(Entry(q, idx))

                for q in queries[::-1]:
                    if q.matched is not None:
                        bodies = list(bodies)
                        bodies.pop(q.matched_pos)
                        if q.node.lhs not in PostOrderDFS(
                            Plan(bodies[q.node_pos + 1 :])
                        ) and isinstance(q.node.rhs, MapJoin | Aggregate | Reorder):
                            bodies[q.node_pos] = Query(
                                q.matched.lhs, Reformat(q.matched.rhs.tns, q.node.rhs)
                            )
                            return Plan(tuple(bodies))
                return None

    return Rewrite(PostWalk(Fixpoint(rule_0)))(root)


@overload
def _propagate_fields(root: Plan, fields: dict[LogicNode, Iterable[Field]]) -> Plan: ...


@overload
def _propagate_fields(
    root: Query, fields: dict[LogicNode, Iterable[Field]]
) -> Query: ...


@overload
def _propagate_fields(
    root: Alias, fields: dict[LogicNode, Iterable[Field]]
) -> Relabel: ...


@overload
def _propagate_fields(
    root: LogicTree, fields: dict[LogicNode, Iterable[Field]]
) -> LogicTree: ...


@overload
def _propagate_fields(
    root: LogicNode, fields: dict[LogicNode, Iterable[Field]]
) -> LogicNode: ...


def _propagate_fields(
    root: LogicNode, fields: dict[LogicNode, Iterable[Field]]
) -> LogicNode:
    match root:
        case Plan(bodies):
            return Plan(tuple(_propagate_fields(b, fields) for b in bodies))
        case Query(lhs, rhs):
            rhs_2 = _propagate_fields(rhs, fields)
            assert isinstance(rhs_2, LogicExpression)
            fields[lhs] = rhs_2.fields
            return Query(lhs, rhs_2)
        case Alias(_) as a:
            return Relabel(a, tuple(fields[a]))
        case LogicTree() as tree:
            return tree.make_term(
                tree.head(), *(_propagate_fields(c, fields) for c in tree.children)
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
                reidx = dict(zip(mj.fields, idxs, strict=True))
                return MapJoin(
                    op,
                    tuple(
                        Relabel(arg, tuple(reidx[f] for f in arg.fields))
                        for arg in args
                    ),
                )

    def rule_1(ex):
        # relabel(agg(..., [1,2,3], 3), [11,22]) =>
        #     agg(..., relabel([1,2,3], [11,22,3]), 3)
        match ex:
            case Relabel(Aggregate(op, init, arg, agg_idxs), relabel_idxs):
                diff_idxs = setdiff(arg.fields, agg_idxs)
                reidx_dict = dict(zip(diff_idxs, relabel_idxs, strict=True))
                relabeled_idxs = tuple(reidx_dict.get(idx, idx) for idx in arg.fields)
                return Aggregate(op, init, Relabel(arg, relabeled_idxs), agg_idxs)

    def rule_2(ex):
        match ex:
            case Relabel(Relabel(arg, _), idxs):
                return Relabel(arg, idxs)

    def rule_3(ex):
        # relabel(reorder(_, [2,1]), [11,22]) => reorder(relabel(_, [22,11]), [11,22])
        match ex:
            case Relabel(Reorder(arg, idxs_1), idxs_2):
                idxs_3 = arg.fields
                reidx_dict = dict(zip(idxs_1, idxs_2, strict=True))
                idxs_4 = tuple(reidx_dict.get(idx, idx) for idx in idxs_3)
                return Reorder(Relabel(arg, idxs_4), idxs_2)

    def rule_4(ex):
        match ex:
            case Relabel(Table(arg, _), idxs):
                return Table(arg, idxs)

    def rule_5(ex):
        match ex:
            case Relabel(Literal() as arg):
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
                            Reorder(arg, intersect(idxs, arg.fields)) for arg in args
                        ),
                    ),
                    idxs,
                )

    def rule_8(ex):
        match ex:
            case Reorder(Aggregate(op, init, arg, idxs_1), idxs_2) if (
                not is_subsequence(intersect(arg.fields, idxs_2), idxs_2)
            ):
                return Reorder(
                    Aggregate(
                        op,
                        init,
                        Reorder(arg, with_subsequence(idxs_2, arg.fields)),
                        idxs_1,
                    ),
                    idxs_2,
                )

    return Rewrite(PreWalk(Chain([Fixpoint(rule_6), rule_7, rule_8])))(root)


def lift_fields(root):
    def rule_0(ex):
        match ex:
            case Aggregate(op, init, arg, idxs):
                return Aggregate(op, init, Reorder(arg, tuple(arg.fields)), idxs)

    def rule_1(ex):
        match ex:
            case Query(lhs, MapJoin() as rhs):
                return Query(lhs, Reorder(rhs, tuple(rhs.fields)))

    def rule_2(ex):
        match ex:
            case Query(lhs, Reformat(tns, MapJoin() as arg)):
                return Query(lhs, Reformat(tns, Reorder(arg, tuple(arg.fields))))

    return Rewrite(PostWalk(Chain([rule_0, rule_1, rule_2])))(root)


def flatten_plans(root):
    def rule_0(ex):
        match ex:
            case Plan(bodies):
                new_bodies = [
                    tuple(body.bodies) if isinstance(body, Plan) else (body,)
                    for body in bodies
                ]
                flatten_bodies = tuple(reduce(lambda x, y: x + y, new_bodies))
                return Plan(flatten_bodies)

    def rule_1(ex):
        match ex:
            case Plan(bodies):
                body_iter = iter(bodies)
                new_bodies = []
                while (body := next(body_iter, None)) is not None:
                    new_bodies.append(body)
                    if isinstance(body, Produces):
                        break
                return Plan(tuple(new_bodies))

    return PostWalk(Fixpoint(Chain([rule_0, rule_1])))(root)


def _propagate_transpose_queries(root, bindings: dict[LogicNode, LogicNode]):
    match root:
        case Plan(bodies):
            return Plan(
                tuple(_propagate_transpose_queries(body, bindings) for body in bodies)
            )
        case Query(lhs, rhs):
            rhs = push_fields(
                Rewrite(PostWalk(lambda node: bindings.get(node, node)))(rhs)
            )
            match rhs:
                case Reorder(Relabel(Alias(_), _), _) | Relabel(Alias(_), _) | Alias(_):
                    bindings[lhs] = rhs
                    return Plan()
                case _:
                    return Query(lhs, rhs)
        case Produces(_) as prod:
            return push_fields(
                Rewrite(PostWalk(lambda node: bindings.get(node, node)))(prod)
            )
        case _:
            raise Exception(f"Invalid node: {root} in propagate_transpose_queries")


def propagate_transpose_queries(root):
    return _propagate_transpose_queries(root, bindings={})


def concordize(root):
    needed_swizzles: dict[Alias, dict[tuple[Field, ...], Alias]] = {}
    namespace = Namespace()
    # update namespace
    unique_leaves: set[Alias | Field] = set()
    for node in PostOrderDFS(root):
        match node:
            case Alias(_) | Field(_):
                unique_leaves.add(node)
    for leaf in unique_leaves:
        namespace.freshen(leaf.name)

    def rule_0(ex):
        match ex:
            case Reorder(Relabel(Alias(_) as alias, idxs_1), idxs_2):
                if not is_subsequence(intersect(idxs_1, idxs_2), idxs_2):
                    idxs_subseq = with_subsequence(intersect(idxs_2, idxs_1), idxs_1)
                    perm = tuple(idxs_1.index(idx) for idx in idxs_subseq)
                    return Reorder(
                        Relabel(
                            needed_swizzles.setdefault(alias, {}).setdefault(
                                perm, Alias(namespace.freshen(alias.name))
                            ),
                            idxs_subseq,
                        ),
                        idxs_2,
                    )
                return None

    def rule_1(ex):
        match ex:
            case Query(lhs, rhs) as q if lhs in needed_swizzles:
                idxs = tuple(rhs.fields)
                swizzle_queries = tuple(
                    Query(
                        alias, Reorder(Relabel(lhs, idxs), tuple(idxs[p] for p in perm))
                    )
                    for perm, alias in needed_swizzles[lhs].items()
                )

                return Plan((q, *swizzle_queries))

    root = flatten_plans(root)
    match root:
        case Plan((*bodies, Produces(_) as prod)):
            root = Plan(tuple(bodies))
            root = Rewrite(PostWalk(rule_0))(root)
            root = Rewrite(PostWalk(rule_1))(root)
            return flatten_plans(Plan((root, prod)))
        case _:
            raise Exception(f"Invalid root: {root}")


def normalize_names(root):
    namespace: Namespace = Namespace()
    scope_dict: dict[str, str] = {}

    def normname(symbol: str) -> str:
        if symbol in scope_dict:
            return scope_dict[symbol]

        if "#" in symbol:
            if (match_obj := re.search(r"##(.+)#\d+", symbol)) or (
                match_obj := re.search(r"#\d+#(.+)", symbol)
            ):
                (new_sym,) = match_obj.groups()
            else:
                raise Exception(f"Invalid symbol: {symbol}")
        else:
            new_sym = symbol

        new_sym = namespace.freshen(new_sym)
        scope_dict[symbol] = new_sym
        return new_sym

    def rule_0(ex):
        match ex:
            case Alias(name):
                return Alias(normname(name))

    def rule_1(ex):
        match ex:
            case Field(name):
                return Field(normname(name))

    root = Rewrite(PostWalk(rule_0))(root)
    return Rewrite(PostWalk(rule_1))(root)


def materialize_squeeze_expand_productions(root):
    def rule_0(ex: LogicNode, preamble: list[Query]):
        match ex:
            case Reorder(Relabel(Alias(_) as tns, idxs_1), idxs_2) if set(
                idxs_1
            ) != set(idxs_2):
                new_tns = Alias(gensym("A"))
                new_idxs = with_subsequence(intersect(idxs_1, idxs_2), idxs_2)
                preamble.append(Query(new_tns, Reorder(Relabel(tns, idxs_1), new_idxs)))
                if new_idxs == idxs_2:
                    return new_tns
                return Reorder(Relabel(new_tns, new_idxs), idxs_2)
            case Reorder(Relabel(arg, idxs_1), idxs_2) if idxs_1 == idxs_2:
                return arg
            case node:
                return node

    def rule_1(ex):
        match ex:
            case Produces(bodies):
                preamble = []
                new_bodies = tuple(rule_0(body, preamble) for body in bodies)
                return Plan(tuple(preamble + [Produces(new_bodies)]))

    return Rewrite(PostWalk(rule_1))(root)


class DefaultLogicOptimizer:
    def __init__(self, ctx: LogicCompiler):
        self.ctx = ctx

    def __call__(self, prgm: LogicNode):
        prgm = optimize(prgm)
        return self.ctx(prgm)
