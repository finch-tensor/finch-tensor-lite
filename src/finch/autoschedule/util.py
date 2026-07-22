from functools import reduce
from typing import overload

from finch.algebra.utils import intersect, is_subsequence, setdiff, with_subsequence
from finch.finch_logic import (
    Aggregate,
    Alias,
    LogicExpression,
    LogicNode,
    LogicStatement,
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    Table,
)
from finch.symbolic import Chain, Fixpoint, PostWalk, PreWalk, Rewrite


@overload
def push_fields(root: LogicExpression) -> LogicExpression: ...
@overload
def push_fields(root: LogicStatement) -> LogicStatement: ...
@overload
def push_fields(root: LogicNode) -> LogicNode: ...
def push_fields(root):
    def rule_1(ex):
        match ex:
            case Relabel(MapJoin(op, args) as mj, idxs):
                reidx = dict(zip(mj.fields(), idxs, strict=True))
                return MapJoin(
                    op,
                    tuple(
                        Relabel(arg, tuple(reidx[f] for f in arg.fields()))
                        for arg in args
                    ),
                )
            case Relabel(Aggregate(op, init, arg, agg_idxs), relabel_idxs):
                diff_idxs = setdiff(arg.fields(), agg_idxs)
                reidx_dict = dict(zip(diff_idxs, relabel_idxs, strict=True))
                relabeled_idxs = tuple(reidx_dict.get(idx, idx) for idx in arg.fields())
                return Aggregate(op, init, Relabel(arg, relabeled_idxs), agg_idxs)
            case Relabel(Relabel(arg, _), idxs):
                return Relabel(arg, idxs)
            case Relabel(Reorder(arg, idxs_1), idxs_2):
                idxs_3 = arg.fields()
                reidx_dict = dict(zip(idxs_1, idxs_2, strict=True))
                idxs_4 = tuple(reidx_dict.get(idx, idx) for idx in idxs_3)
                return Reorder(Relabel(arg, idxs_4), idxs_2)
            case Relabel(Table(arg, _), idxs):
                return Table(arg, idxs)

    root = Rewrite(PreWalk(Fixpoint(rule_1)))(root)  # ignore[type-arg]

    def rule_2(ex):
        match ex:
            case Reorder(Reorder(arg, _), idxs):
                return Reorder(arg, idxs)
            case Reorder(MapJoin(op, args), idxs) if not all(
                isinstance(arg, Reorder) and is_subsequence(arg.fields(), idxs)
                for arg in args
            ):
                return Reorder(
                    MapJoin(
                        op,
                        tuple(
                            Reorder(arg, intersect(idxs, arg.fields())) for arg in args
                        ),
                    ),
                    idxs,
                )
            case Reorder(Aggregate(op, init, arg, idxs_1), idxs_2) if (
                not is_subsequence(intersect(arg.fields(), idxs_2), idxs_2)
            ):
                return Reorder(
                    Aggregate(
                        op,
                        init,
                        Reorder(arg, with_subsequence(idxs_2, arg.fields())),
                        idxs_1,
                    ),
                    idxs_2,
                )

    return Rewrite(PreWalk(Fixpoint(rule_2)))(root)


def flatten_plans(root):
    def rule_0(ex):
        match ex:
            case Plan(bodies):
                new_bodies = [
                    tuple(body.bodies) if isinstance(body, Plan) else (body,)
                    for body in bodies
                ]
                flatten_bodies = tuple(reduce(lambda x, y: x + y, new_bodies, ()))
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


def propagate_copy_queries(root, bindings):
    copies = {}

    def rule_1(node):
        match node:
            case Query(lhs, Table(Alias(_) as rhs, _)) if lhs not in bindings:
                copies[lhs] = copies.get(rhs, rhs)
                return Plan()
            case Query(lhs, Reorder(Table(Alias(_) as rhs, idxs_1), idxs_2)) if (
                idxs_1 == idxs_2 and lhs not in bindings
            ):
                copies[lhs] = copies.get(rhs, rhs)
                return Plan()

    root = Rewrite(PostWalk(rule_1))(root)

    def rule_2(ex):
        match ex:
            case Alias() as a if a in copies:
                return copies[a]

    return Rewrite(PostWalk(rule_2))(root)
