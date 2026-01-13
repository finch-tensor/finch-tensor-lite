from functools import reduce
from itertools import chain as join_chains
from typing import overload

from finchlite.algebra.algebra import is_annihilator, is_distributive, is_identity
from finchlite.algebra.tensor import TensorFType
from finchlite.finch_assembly.stages import AssemblyLibrary
from finchlite.finch_logic.nodes import LogicExpression
from finchlite.finch_logic.stages import LogicLoader
from finchlite.symbolic import gensym

from ..finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
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
from ..symbolic import (
    Fixpoint,
    Namespace,
    PostOrderDFS,
    PostWalk,
    PreWalk,
    Rewrite,
)
from ._utils import intersect, setdiff
from .standardize import (
    concordize,
    flatten_plans,
    isolate_aggregates,
    push_fields,
)


def with_unique_lhs(
    f, root: LogicStatement, bindings: dict[Alias, TensorFType]
) -> tuple[LogicStatement, dict[Alias, TensorFType]]:
    spc = Namespace(root)
    for var in bindings:
        spc.freshen(var.name)
    renames: dict[Alias, Alias] = {}
    bound = set(bindings.keys())
    writes: dict[Alias, Alias] = {}

    def rule_0(node):
        match node:
            case Query(lhs, rhs):
                if lhs in bound:
                    var = Alias(spc.freshen(lhs.name))
                    renames[lhs] = var
                    if lhs in bindings:
                        writes[lhs] = var
                    return Query(var, rhs)
                bound.add(lhs)
                return None
            case Alias() as a if a in renames:
                return renames[a]
            case Produces(args):
                return Produces(args + tuple(writes.values()))

    root = Rewrite(PostWalk(rule_0))(root)
    root, bindings = f(root, bindings)

    def rule_1(node):
        match node:
            case Produces(args):
                bodies: list[LogicStatement] = []
                for k, v in writes.items():
                    idxs = tuple(
                        Field(spc.freshen("i")) for _ in range(bindings[k].ndim)
                    )
                    bodies.append(Query(v, Table(k, idxs)))
                return Plan(tuple(bodies) + (Produces(args[: -len(writes) + 1]),))

    return (Rewrite(PostWalk(rule_1))(root), bindings)


def optimize(
    prgm: LogicStatement, bindings: dict[Alias, TensorFType]
) -> tuple[LogicStatement, dict[Alias, TensorFType]]:
    def transform(prgm, bindings):
        prgm = push_fields(prgm)
        prgm = propagate_single_use_queries_and_combine_aggregates(prgm)

        prgm = isolate_aggregates(prgm)

        prgm = propagate_copy_queries(prgm)
        prgm = propagate_transpose_queries(prgm)
        prgm = propagate_map_queries(prgm)

        prgm = push_fields(prgm)
        prgm = lift_fields(prgm)
        prgm = push_fields(prgm)

        prgm = propagate_transpose_queries(prgm)
        prgm = push_fields(prgm)
        prgm = set_loop_order(prgm)
        prgm = push_fields(prgm)

        prgm = concordize(prgm)

        return propagate_copy_queries(prgm), bindings

    return with_unique_lhs(transform, prgm, bindings)


def get_productions(root: LogicStatement) -> tuple[Alias, ...]:
    match root:
        case Plan(bodies):
            return get_productions(bodies[-1])
        case Produces(args):
            return args
        case Query(lhs, _):
            return (lhs,)
        case _:
            raise ValueError(f"Invalid node type: {type(root)}")


def propagate_map_queries(root: LogicStatement) -> LogicStatement:
    def rule_0(ex):
        match ex:
            case Aggregate(op, init, arg, ()):
                return MapJoin(op, (init, arg))

    root = Rewrite(PostWalk(rule_0))(root)
    rets = get_productions(root)
    props = {}

    def rule_1(node):
        match node:
            case Query(a, MapJoin(op, args)) if a not in rets:
                props[a] = MapJoin(op, args)
                return Plan()
            case Table(a, idxs) if a in props:
                return Relabel(props[a], idxs)

    root = Rewrite(PostWalk(rule_1))(root)
    return flatten_plans(root)


def propagate_single_use_queries_and_combine_aggregates(
    root: LogicStatement,
) -> LogicStatement:
    def rule_0(ex):
        match ex:
            case Aggregate(op, init, arg, ()):
                return MapJoin(op, (init, arg))

    root = Rewrite(PostWalk(rule_0))(root)

    uses: dict[LogicNode, int] = {}
    defs: dict[LogicNode, LogicNode] = {}
    for node in PostOrderDFS(root):
        match node:
            case Alias() as a:
                uses[a] = uses.get(a, 0) + 1
            case Query(a, b):
                uses[a] = uses.get(a, 0) - 1
                defs[a] = b

    rets = get_productions(root)

    def rule_1(ex):
        match ex:
            case Query(a, _) if uses[a] == 1 and a not in rets:
                return Plan()
            case Table(Alias() as a, idxs) if (
                uses.get(a, 0) == 1 and a not in rets and a in defs
            ):
                return Relabel(defs[a], idxs)

    root = Rewrite(PreWalk(rule_1))(root)
    root = push_fields(root)

    def rule_2(ex):
        match ex:
            case MapJoin(
                Literal(f),
                args,
            ):
                for idx, item in reversed(list(enumerate(args))):
                    before_item = args[:idx]
                    after_item = args[idx + 1 :]
                    match item:
                        case Aggregate(Literal(g), Literal(init), arg, idxs) as agg if (
                            is_distributive(f, g)
                            and is_annihilator(f, init)
                            and len(agg.fields())
                            == len(
                                MapJoin(
                                    Literal(f), (*before_item, *after_item)
                                ).fields()
                            )
                        ):
                            return Aggregate(
                                Literal(g),
                                Literal(init),
                                MapJoin(Literal(f), (*before_item, arg, *after_item)),
                                idxs,
                            )
            case Aggregate(
                Literal() as op_1,
                Literal() as init_1,
                Aggregate(op_2, Literal() as init_2, arg, idxs_1),
                idxs_2,
            ) if op_1 == op_2 and is_identity(op_2.val, init_2.val):
                return Aggregate(op_1, init_1, arg, idxs_1 + idxs_2)
            case Aggregate(
                Literal() as op_1,
                Literal() as init_1,
                Reorder(Aggregate(op_2, Literal() as init_2, arg, idxs_1), idxs_3),
                idxs_2,
            ) if op_1 == op_2 and is_identity(op_2.val, init_2.val):
                return Reorder(
                    Aggregate(op_1, init_1, arg, idxs_1 + idxs_2),
                    setdiff(idxs_3, idxs_2),
                )

        return None

    return Rewrite(Fixpoint(PreWalk(rule_2)))(root)


def propagate_copy_queries(root):
    copies = {}

    def rule_0(node):
        match node:
            case Query(lhs, Table(Alias(_) as rhs, _)):
                copies[lhs] = copies.get(rhs, rhs)
                return Plan()
            case Query(lhs, Reorder(Table(Alias(_) as rhs, idxs_1), idxs_2)) if (
                idxs_1 == idxs_2
            ):
                copies[lhs] = copies.get(rhs, rhs)
                return Plan()

    root = Rewrite(PostWalk(rule_0))(root)

    def rule_1(ex):
        match ex:
            case Alias() as a if a in copies:
                return copies[a]

    return Rewrite(PostWalk(rule_1))(root)


def lift_fields(root):
    def rule_0(ex):
        match ex:
            case Aggregate(op, init, arg, idxs):
                return Aggregate(op, init, Reorder(arg, tuple(arg.fields())), idxs)
            case Query(lhs, MapJoin() as rhs):
                return Query(lhs, Reorder(rhs, tuple(rhs.fields())))

    return Rewrite(PostWalk(rule_0))(root)


def propagate_transpose_queries(root: LogicStatement):
    props: dict[Alias, LogicExpression] = {}

    def rule_1(node):
        match node:
            case Table(Alias() as a, idxs) if a in props:
                return Relabel(props[a], idxs)
            case Produces(args):
                bodies = [Query(a, props[a]) for a in args if a in props]
                return Plan(tuple(bodies) + (Produces(args),))

    def rule_0(node):
        match node:
            case Query(lhs, Table(Alias(_), _) as rhs):
                props[lhs] = Rewrite(PostWalk(rule_1))(rhs)
                return Plan()
            case Query(lhs, Reorder(Table(Alias(_), _), _) as rhs):
                props[lhs] = Rewrite(PostWalk(rule_1))(rhs)
                return Plan()

    root = push_fields(root)

    root = Rewrite(PostWalk(rule_0))(root)

    root = Rewrite(PostWalk(rule_1))(root)

    return flatten_plans(push_fields(root))


def toposort(chains: list[list[Field]]) -> tuple[Field, ...]:
    chains = [c for c in chains if len(c) > 0]
    parents = {chain[0]: 0 for chain in chains}
    for chain in chains:
        for f in chain[1:]:
            parents[f] = parents.get(f, 0) + 1
    roots = [f for f in parents if parents[f] == 0]
    perm = []
    while len(parents) > 0:
        if len(roots) == 0:
            raise Exception("Cycle detected in fields' orders")
        perm.append(roots.pop())
        for chain in chains:
            if len(chain) > 0 and chain[0] == perm[-1]:
                chain.pop(0)
                if len(chain) > 0:
                    parents[chain[0]] -= 1
                    if parents[chain[0]] == 0:
                        roots.append(chain[0])
        parents.pop(perm[-1])
    return tuple(perm)


def _heuristic_loop_order(root: LogicExpression) -> tuple[Field, ...]:
    chains = []
    for node in PostOrderDFS(root):
        match node:
            case Reorder(Table(_, idxs_1), idxs_2):
                chains.append(list(intersect(intersect(idxs_1, idxs_2), root.fields())))
    chains.extend([f] for f in root.fields())
    result = toposort(chains)
    if reduce(max, [len(c) for c in chains], 0) < len(set(join_chains(*chains))):
        counts: dict[Field, int] = {}
        for chain in chains:
            for f in chain:
                counts[f] = counts.get(f, 0) + 1
        result = tuple(sorted(result, key=lambda x: counts[x] == 1))
    return result


@overload
def _set_loop_order(
    node: LogicStatement, perms: dict[LogicNode, LogicExpression]
) -> LogicStatement: ...
@overload
def _set_loop_order(
    node: LogicNode, perms: dict[LogicNode, LogicExpression]
) -> LogicNode: ...
def _set_loop_order(node, perms):
    def rule_0(node):
        match node:
            case Table(Alias(_) as tns, idxs) if tns in perms:
                return Relabel(perms[tns], idxs)
        return None

    match node:
        case Plan(bodies):
            return Plan(tuple(_set_loop_order(body, perms) for body in bodies))
        case Query(lhs, Aggregate(op, init, arg, idxs) as rhs):
            rhs = push_fields(Rewrite(PostWalk(rule_0))(rhs))
            assert isinstance(arg, LogicExpression)
            idxs_2 = _heuristic_loop_order(arg)
            rhs_2 = Aggregate(op, init, Reorder(arg, idxs_2), idxs)
            perms[lhs] = Reorder(Table(lhs, tuple(rhs_2.fields())), tuple(rhs.fields()))
            return Query(lhs, rhs_2)
        case Query(lhs, Reorder(Table(Alias(_) as tns, _), idxs)) as q:
            tns = perms.get(tns, tns)
            perms[lhs] = Table(lhs, idxs)
            return q
        case Query(lhs, rhs):  # assuming rhs is a bunch of mapjoins
            rhs = push_fields(Rewrite(PostWalk(rule_0))(rhs))
            assert isinstance(rhs, LogicExpression)
            idxs = _heuristic_loop_order(rhs)
            perms[lhs] = Reorder(Table(lhs, idxs), tuple(rhs.fields()))
            rhs_2 = Reorder(rhs, idxs)
            return Query(lhs, rhs_2)
        case Produces(args):
            renames = {a: Alias(gensym("A")) for a in args if a in perms}
            bodies = tuple([Query(v, perms[k]) for k, v in renames.items()])
            args_2 = tuple([renames.get(a, a) for a in args])
            return Plan(bodies + (Produces(args_2),))
        case _:
            raise Exception(f"Invalid node: {node} in set_loop_order")


def set_loop_order(node: LogicNode) -> LogicNode:
    return _set_loop_order(node, {})


class DefaultLogicOptimizer(LogicLoader):
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(
        self, prgm: LogicStatement, bindings: dict[Alias, TensorFType]
    ) -> tuple[
        AssemblyLibrary, dict[Alias, TensorFType], dict[Alias, tuple[Field | None, ...]]
    ]:
        prgm, bindings = optimize(prgm, bindings)
        return self.ctx(prgm, bindings)
