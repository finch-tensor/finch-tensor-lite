from functools import reduce
from itertools import chain as join_chains
from typing import overload

from finchlite.algebra import ffuncs
from finchlite.algebra.algebra import is_annihilator, is_distributive, is_identity
from finchlite.algebra.tensor import TensorFType
from finchlite.algebra.utils import intersect, setdiff
from finchlite.autoschedule.stages import LogicFusionOptimizer
from finchlite.finch_logic import (
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
    StatsFactory,
    Table,
    TensorStats,
)
from finchlite.finch_logic.nodes import LogicExpression
from finchlite.symbolic import (
    Fixpoint,
    Namespace,
    PostOrderDFS,
    PostWalk,
    PreWalk,
    Rewrite,
    gensym,
)

from .standardize import (
    concordize,
    flatten_plans,
    isolate_aggregates,
    push_fields,
    wrap_bare_table_queries,
)


def with_unique_lhs(
    f, root: LogicStatement, bindings: dict[Alias, TensorFType]
) -> tuple[LogicStatement, dict[Alias, TensorFType]]:
    """
    Ensures all left-hand sides (LHS) of queries are unique by inserting new
    tensors.
    """

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

    unrenames = {v: k for k, v in renames.items()}

    # Some produces may be renamed during the optimization,
    # so we need to be a bit careful here.
    def rule_1(node):
        match node:
            case Produces(args):
                n_writes = len(writes)
                v_post_list = args[-n_writes:] if n_writes else ()

                bodies: list[LogicStatement] = []
                for (k, _v_pre), v_post in zip(
                    writes.items(), v_post_list, strict=True
                ):
                    idxs = tuple(
                        Field(spc.freshen("i")) for _ in range(bindings[k].ndim)
                    )
                    bodies.append(Query(k, Table(v_post, idxs)))

                v_post_to_k = dict(zip(v_post_list, writes.keys(), strict=True))
                args_2 = tuple(
                    v_post_to_k.get(a, unrenames.get(a, a))
                    for a in args[: len(args) - n_writes]
                )
                return Plan(tuple(bodies) + (Produces(args_2),))

    return (Rewrite(PostWalk(rule_1))(root), bindings)


def add_aggregates(
    root: LogicStatement, bindings: dict[Alias, TensorFType]
) -> LogicStatement:
    fill_values = root.infer_fill_value(
        {var: val.fill_value for var, val in bindings.items()}
    )

    def rule_0(node):
        match node:
            case Query(lhs, Reorder(Aggregate(_, _, arg, idxs), _)):
                return node
            case Query(lhs, Reorder(arg, idxs)):
                return Query(
                    lhs,
                    Reorder(
                        Aggregate(
                            Literal(ffuncs.overwrite),
                            Literal(fill_values[lhs]),
                            arg,
                            (),
                        ),
                        idxs,
                    ),
                )
            case Query(lhs, Aggregate(_, _, arg, idxs)):
                return node
            case Query(lhs, arg):
                return Query(
                    lhs,
                    Aggregate(
                        Literal(ffuncs.overwrite), Literal(fill_values[lhs]), arg, ()
                    ),
                )

    return Rewrite(PostWalk(rule_0))(root)


def optimize(
    prgm: LogicStatement, bindings: dict[Alias, TensorFType]
) -> tuple[LogicStatement, dict[Alias, TensorFType]]:
    def transform(prgm, bindings):
        prgm = push_fields(prgm)
        prgm = propagate_map_queries_backward(prgm)

        prgm = isolate_aggregates(prgm)

        prgm = propagate_copy_queries(prgm)
        prgm = propagate_transpose_queries(prgm)
        prgm = propagate_map_queries(prgm)

        prgm = push_fields(prgm)
        prgm = lift_fields(prgm)
        prgm = push_fields(prgm)

        prgm = propagate_transpose_queries(prgm)
        prgm = push_fields(prgm)
        # prgm = set_loop_order(prgm)
        # prgm = push_fields(prgm)

        prgm = concordize(prgm, bindings)
        prgm = propagate_copy_queries(prgm)
        prgm = add_aggregates(prgm, bindings)
        return prgm, bindings

    prgm, bindings = with_unique_lhs(transform, prgm, bindings)
    return flatten_plans(prgm), bindings


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
    # TODO: We're not ready for that optimization yet.
    #       First we need to support Literals as MapJoin's
    #       arguments (they're missing `fields`) or promote
    #       `init` to a Table here.
    # def rule_0(ex):
    #     match ex:
    #         case Aggregate(op, init, arg, ()):
    #             return MapJoin(op, (init, arg))
    # root = Rewrite(PostWalk(rule_0))(root)

    assert isinstance(root, LogicNode)
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


def propagate_map_queries_backward(root: LogicStatement) -> LogicStatement:
    # TODO: We're not ready for that optimization yet.
    #       First we need to support Literals as MapJoin's
    #       arguments (they're missing `fields`) or promote
    #       `init` to a Table here.
    # def rule_0(ex):
    #     match ex:
    #         case Aggregate(op, init, arg, ()):
    #             return MapJoin(op, (init, arg))
    # root = Rewrite(PostWalk(rule_0))(root)

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


# Everything down here moved to loop_ordering.py
class CycleInFields(Exception): ...


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
            raise CycleInFields("Cycle detected in fields' orders")
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

    need_fix = False
    try:
        result = toposort(chains)
    except CycleInFields:
        import warnings

        warnings.warn("Cycle in fields detected, need to permute.", stacklevel=1)
        need_fix = True
        result = root.fields()

    if need_fix or reduce(max, [len(c) for c in chains], 0) < len(
        set(join_chains(*chains))
    ):
        counts: dict[Field, int] = {}
        for chain in chains:
            for f in chain:
                counts[f] = counts.get(f, 0) + 1
        result = tuple(sorted(result, key=lambda x: counts[x] == 1))
    return result


def _align(
    ex: LogicExpression,
    loop_order: tuple[Field, ...],
    bindings: dict[Alias, TensorFType],
    namespace: Namespace,
) -> tuple[LogicExpression, tuple[Query, ...]]:
    """Align each ``Table`` / ``Reorder(Table, ...)`` to ``loop_order``."""
    needed_swizzles: dict[
        tuple[Alias, tuple[Field, ...], tuple[Field, ...]], Alias
    ] = {}

    def rule(node: LogicNode) -> LogicNode | None:
        match node:
            case Reorder(Table(Alias() as var, physical), logical):
                is_reorder = True
            case Table(Alias() as var, physical):
                logical = physical
                is_reorder = False
            case _:
                return None

        field_set = set(logical)
        desired = tuple(f for f in loop_order if f in field_set)
        if desired == logical:
            if is_reorder and physical == logical:
                return Table(var, physical)
            if is_reorder:
                key = (var, physical, desired)
                if key not in needed_swizzles:
                    needed_swizzles[key] = Alias(namespace.freshen(var.name))
                return Table(needed_swizzles[key], desired)
            return None
        if len(desired) != len(physical) or set(desired) != set(physical):
            return None

        key = (var, physical, desired)
        if key not in needed_swizzles:
            needed_swizzles[key] = Alias(namespace.freshen(var.name))
        return Table(needed_swizzles[key], desired)

    new_ex = Rewrite(PostWalk(rule))(ex)
    queries = tuple(
        Query(alias, Reorder(Table(var, physical), desired))
        for (var, physical, desired), alias in needed_swizzles.items()
    )
    return new_ex, queries

def output_ordered_aggregate(rhs: LogicExpression):
    # Unwrap Reorder(...)/Aggregate(...) to find an output-ordered aggregate.
    output_order = None
    while True:
        match rhs:
            case Reorder(arg, order):
                output_order = order
                rhs = arg
            case Aggregate(op, init, arg, reduce_axes) if (
                output_order is not None
            ):
                return op, init, arg, reduce_axes, output_order
            case _:
                return None

def strip_noop_reorders(ex: LogicExpression) -> LogicExpression:
# Drop Reorder nodes whose order already matches the arg's fields.
    def rule(node: LogicNode) -> LogicNode | None:
        match node:
            case Reorder(arg, order) if arg.fields() == order:
                return arg
            case _:
                return None

    return Rewrite(PostWalk(rule))(ex)

def set_loop_order(node: LogicStatement) -> LogicStatement:
    def with_swizzles(
        ordered: Query, swizzles: tuple[Query, ...]
    ) -> LogicStatement:
        if swizzles:
            return Plan((*swizzles, ordered))
        return ordered

    def align_aggregate(
        op: Literal,
        init: Literal,
        arg: LogicExpression,
        reduce_axes: tuple[Field, ...],
    ) -> tuple[Aggregate, tuple[Query, ...]]:
        # This case preserves an existing aggregate loop order
        # from Logic Optimizer.
        match arg:
            case Reorder(inner, old_loop_order):
                arg = inner
                loop_order = old_loop_order
            case _:
                loop_order = tuple(arg.fields())
        arg, swizzles = _align(arg, loop_order, bindings, namespace)
        arg = strip_noop_reorders(arg)
        return Aggregate(
            op, init, Reorder(arg, loop_order), reduce_axes
        ), swizzles

    match node:
        case Plan(bodies):
            return Plan(tuple(set_loop_order(body) for body in bodies))
        case Query(lhs, rhs) if (
            output_aggregate := output_ordered_aggregate(rhs)
        ) is not None:
            op, init, arg, reduce_axes, output_order = output_aggregate
            aggregate, swizzles = align_aggregate(op, init, arg, reduce_axes)
            return with_swizzles(
                Query(lhs, Reorder(aggregate, output_order)), swizzles
            )
        case Query(lhs, Table(_, idxs) as rhs):
            return Query(lhs, Reorder(rhs, idxs))
        case Query(lhs, Reorder(arg, output_order)):
            arg, swizzles = _align(arg, output_order, bindings, namespace)
            arg = strip_noop_reorders(arg)
            return with_swizzles(
                Query(lhs, Reorder(arg, output_order)), swizzles
            )
        case Query(lhs, Aggregate(op, init, arg, reduce_axes)):
            aggregate, swizzles = align_aggregate(op, init, arg, reduce_axes)
            return with_swizzles(Query(lhs, aggregate), swizzles)
        case Query(lhs, rhs):
            loop_order = self.get_loop_order(
                node, bindings, stats, stats_factory
            )
            rhs, swizzles = _align(rhs, loop_order, bindings, namespace)
            rhs = strip_noop_reorders(rhs)
            match rhs:
                case Aggregate(op, init, arg, reduce_axes):
                    aggregate, swizzles = align_aggregate(
                        op, init, arg, reduce_axes
                    )
                    return with_swizzles(Query(lhs, aggregate), swizzles)
                case _:
                    return with_swizzles(
                        Query(lhs, Reorder(rhs, loop_order)), swizzles
                    )
        case Produces(_):
            return node
        case _:
            raise ValueError(
                f"Unsupported logic statement for loop ordering: {node}"
            )


class DefaultLogicOptimizer(LogicFusionOptimizer):
    def __init__(self, ctx):
        self.ctx = ctx

    def lower(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, "TensorStats"],
        stats_factory: StatsFactory,
    ):
        prgm, bindings = optimize(prgm, bindings)
        prgm = wrap_bare_table_queries(prgm)
        return self.ctx(prgm, bindings, stats, stats_factory)
