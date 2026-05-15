from functools import reduce
from typing import overload

from finchlite.algebra import ffuncs, is_associative, is_commutative
from finchlite.algebra.tensor import TensorFType
from finchlite.algebra.utils import intersect, is_subsequence, setdiff, with_subsequence
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicExpression,
    LogicLoader,
    LogicNode,
    LogicStatement,
    MapJoin,
    MockLogicLoader,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    StatsFactory,
    Table,
    TensorStats,
)
from finchlite.symbolic import (
    Chain,
    Fixpoint,
    Namespace,
    PostOrderDFS,
    PostWalk,
    PreWalk,
    Rewrite,
    gensym,
)

from .normalize import normalize_names


def isolate_aggregates(root: LogicStatement) -> LogicStatement:
    def transform(stmt):
        stack = []

        def rule_1(ex):
            match ex:
                case Aggregate(_, _, _, _) as agg:
                    var = Alias(gensym("A"))
                    stack.append(Query(var, agg))
                    return Table(var, agg.fields())
                case _:
                    return None

        match stmt:
            case Query(lhs, Aggregate(op, init, arg, idxs)):
                arg = Rewrite(PostWalk(rule_1))(arg)
                return Plan((*stack, Query(lhs, Aggregate(op, init, arg, idxs))))
            case Query(lhs, rhs):
                rhs = Rewrite(PostWalk(rule_1))(rhs)
                return Plan((*stack, Query(lhs, rhs)))
            case Produces(args):
                args = tuple(Rewrite(PostWalk(rule_1))(arg) for arg in args)
                return Plan((*stack, Produces(args)))
            case _:
                return None

    return Rewrite(PostWalk(transform))(root)


def standardize_inplace_queries(
    root: LogicStatement, bindings: dict[Alias, TensorFType]
) -> LogicStatement:
    fill_values = root.infer_fill_value(
        {var: val.fill_value for var, val in bindings.items()}
    )

    # Flatten map joins with the same associative operator as root mapjoin
    def rule_1(stmt):
        match stmt:
            case Reorder(MapJoin(Literal(op), args), idxs) if is_associative(op):
                new_args = []
                for arg in args:
                    match arg:
                        case Reorder(MapJoin(Literal(op_1), args_1), _) if op_1 == op:
                            new_args.extend(args_1)
                        case _:
                            new_args.append(arg)

                return Reorder(MapJoin(Literal(op), tuple(new_args)), idxs)

    root = Rewrite(PostWalk(rule_1))(root)

    # Reorder operands of commutative mapjoins to have inplace operand first
    def rule_2(stmt):
        match stmt:
            case Query(lhs, Reorder(MapJoin(op, args), idxs_1)) if is_commutative(
                op.val
            ):
                lhs_arg = list(filter(lambda arg: lhs in PostOrderDFS(arg), args))
                if len(lhs_arg) != 1:
                    return None

                lhs_arg = lhs_arg[0]
                match lhs_arg:
                    case Reorder(Table(_), idxs_2) if idxs_2 == idxs_1:
                        others = filter(lambda arg: arg is not lhs_arg, args)
                        return Query(
                            lhs,
                            Reorder(MapJoin(op, (lhs_arg, *others)), idxs_1),
                        )

                    case _:
                        return None

    root = Rewrite(PostWalk(rule_2))(root)

    # Unflatten mapjoins since most operators are not nary
    def rule_3(stmt):
        match stmt:
            case Reorder(MapJoin(op, args), idxs) if len(args) > 2 and is_associative(
                op.val
            ):
                return Reorder(
                    MapJoin(op, (args[0], Reorder(MapJoin(op, args[1:]), idxs))), idxs
                )

    root = Rewrite(PreWalk(rule_3))(root)

    # Wrap relevant mapjoin operands in aggregate to conform with normal form
    def rule_4(stmt):
        match stmt:
            case Query(
                lhs,
                Reorder(
                    MapJoin(
                        op,
                        (
                            Reorder(Table(lhs_1), idxs_1) as lhs_arg,
                            Reorder(MapJoin(), _) as non_lhs_arg,
                        ),
                    ),
                    idxs_2,
                ),
            ) if lhs_1 == lhs and idxs_1 == idxs_2:
                return Query(
                    lhs,
                    Reorder(
                        MapJoin(
                            op,
                            (
                                lhs_arg,
                                Aggregate(
                                    Literal(ffuncs.overwrite),
                                    Literal(non_lhs_arg.fill_value(fill_values)),
                                    non_lhs_arg,
                                    (),
                                ),
                            ),
                        ),
                        idxs_2,
                    ),
                )

    return Rewrite(PostWalk(rule_4))(root)


def split_increments(root: LogicStatement) -> LogicStatement:
    def rule_2(stmt):
        match stmt:
            case Query(
                lhs,
                Reorder(
                    MapJoin(op, (Reorder(Table(lhs_1), idxs_1), Reorder(Table(), _))),
                    idxs_2,
                ),
            ) if lhs_1 == lhs and idxs_1 == idxs_2:
                return None
            case Query(
                lhs,
                Reorder(
                    MapJoin(
                        Literal(op),
                        (
                            Reorder(Table(lhs_1), idxs_1),
                            Aggregate(Literal(op_1), _, _, _),
                        ),
                    ),
                    idxs_2,
                ),
            ) if lhs_1 == lhs and idxs_1 == idxs_2 and op_1 in (op, ffuncs.overwrite):
                return None
            case Query(lhs, rhs):
                if lhs in PostOrderDFS(rhs):
                    var = Alias(gensym("A"))
                    new_query = Query(var, rhs)
                    new_root = Query(lhs, Table(var, rhs.fields()))
                    return Plan((new_query, new_root))
        return None

    return Rewrite(PostWalk(rule_2))(root)


def standardize_query_roots(
    root: LogicStatement, bindings: dict[Alias, TensorFType]
) -> LogicStatement:
    fill_values = root.infer_fill_value(
        {var: val.fill_value for var, val in bindings.items()}
    )

    def rule(ex):

        match ex:
            case Query(
                lhs,
                Aggregate(op, init, Reorder(arg, idxs_1), idxs_2) as rhs,
            ):
                return ex
            case Query(lhs, Aggregate(op, init, arg, idxs_2) as rhs):
                idxs_1 = arg.fields()
                return Query(lhs, Aggregate(op, init, Reorder(arg, idxs_1), idxs_2))
            case Query(lhs, Reorder(Table(Alias(), idxs_1), idxs_2)):
                return ex
            case Query(lhs, Table(Alias(), idxs) as arg):
                return Query(lhs, Reorder(arg, idxs))
            case Query(
                lhs,
                Reorder(
                    MapJoin(op, (Reorder(Table(lhs_1), idxs_1), Reorder(Table(), _))),
                    idxs_2,
                ),
            ) if lhs_1 == lhs and idxs_1 == idxs_2:
                return ex
            case Query(
                lhs,
                Reorder(
                    MapJoin(
                        Literal(op),
                        (
                            Reorder(Table(lhs_1), idxs_1),
                            Aggregate(Literal(op_1), _, _, _),
                        ),
                    ),
                    idxs_2,
                ),
            ) if lhs_1 == lhs and idxs_1 == idxs_2 and op_1 in (op, ffuncs.overwrite):
                return ex
            case Query(lhs, rhs):
                return Query(
                    lhs,
                    Aggregate(
                        Literal(ffuncs.overwrite),
                        Literal(rhs.fill_value(fill_values)),
                        Reorder(rhs, rhs.fields()),
                        (),
                    ),
                )

    return Rewrite(PostWalk(rule))(root)


def concordize(
    root: LogicStatement, bindings: dict[Alias, TensorFType]
) -> LogicStatement:
    needed_swizzles: dict[Alias, dict[tuple[int, ...], Alias]] = {}
    namespace = Namespace(root)

    def rule_0(ex):
        match ex:
            case Reorder(Table(Alias(_) as var, idxs_1), idxs_2):
                if not is_subsequence(intersect(idxs_1, idxs_2), idxs_2):
                    idxs_subseq = with_subsequence(intersect(idxs_2, idxs_1), idxs_1)
                    perm = tuple(idxs_1.index(idx) for idx in idxs_subseq)
                    return Reorder(
                        Table(
                            needed_swizzles.setdefault(var, {}).setdefault(
                                perm, Alias(namespace.freshen(var.name))
                            ),
                            idxs_subseq,
                        ),
                        idxs_2,
                    )
                return None

    def _get_swizzle_queries(lhs: Alias) -> tuple[Query, ...]:
        ndims = len(next(iter(needed_swizzles[lhs].items()))[0])
        idxs = tuple([Field(f"i_{i}") for i in range(ndims)])
        return tuple(
            Query(alias, Reorder(Table(lhs, idxs), tuple(idxs[p] for p in perm)))
            for perm, alias in needed_swizzles[lhs].items()
        )

    def rule_1(ex):
        match ex:
            case Query(lhs, _) as q if lhs in needed_swizzles:
                swizzle_queries = _get_swizzle_queries(lhs)
                return Plan((q, *swizzle_queries))

    root = flatten_plans(root)
    match root:
        case Plan((*bodies, Produces(_) as prod)):
            root = Plan(tuple(bodies))
            root = Rewrite(PostWalk(rule_0))(root)
            root = Rewrite(PostWalk(rule_1))(root)
            # Consider also aliases from input arguments.
            for alias in bindings:
                if alias in needed_swizzles:
                    swizzle_queries = _get_swizzle_queries(alias)
                    root = Plan((*swizzle_queries, root))
            return flatten_plans(Plan((root, prod)))
        case _:
            raise Exception(f"Invalid root: {root}")


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


def drop_reorders(root: LogicStatement) -> LogicStatement:
    def rule_2(stmt):
        match stmt:
            case Query(lhs, Aggregate(op, init, Reorder(arg, idxs_1), idxs_2)):

                def rule(ex):
                    match ex:
                        case Reorder(arg_2, idxs_3):
                            assert is_subsequence(idxs_3, idxs_1)
                            return arg_2

                arg_3 = Rewrite(PostWalk(rule))(arg)
                return Query(lhs, Aggregate(op, init, Reorder(arg_3, idxs_1), idxs_2))

            case Query(lhs, Reorder(MapJoin(op, args), idxs)):

                def rule(ex):
                    match ex:
                        case Reorder(
                            Aggregate(_, _, Reorder(_, idxs_2), idxs_3) as agg_arg,
                            idxs_4,
                        ):
                            assert (
                                tuple(filter(lambda x: x not in idxs_3, idxs_2))
                                == idxs_4
                            )
                            return agg_arg

                arg_3 = tuple(Rewrite(PostWalk(rule))(arg) for arg in args)
                return Query(lhs, Reorder(MapJoin(op, arg_3), idxs))

    return Rewrite(PostWalk(rule_2))(root)


def drop_with_aggregation(root: LogicStatement) -> LogicStatement:
    def rule_2(stmt):
        match stmt:
            case Query(lhs, Aggregate(op, init, Reorder(arg, idxs_1), idxs_2)):
                idxs_3 = tuple([idx for idx in arg.fields() if idx not in idxs_1])
                return Query(
                    lhs,
                    Aggregate(op, init, Reorder(arg, idxs_1 + idxs_3), idxs_2 + idxs_3),
                )

    return Rewrite(PostWalk(rule_2))(root)


def standardize(
    prgm: LogicStatement,
    bindings: dict[Alias, TensorFType],
) -> tuple[LogicStatement, dict[Alias, TensorFType]]:
    prgm = isolate_aggregates(prgm)
    prgm = push_fields(prgm)
    prgm = drop_reorders(prgm)
    prgm = standardize_inplace_queries(prgm, bindings)
    prgm = split_increments(prgm)
    prgm = standardize_query_roots(prgm, bindings)
    prgm = push_fields(prgm)
    prgm = drop_reorders(prgm)
    prgm = drop_with_aggregation(prgm)
    prgm = concordize(prgm, bindings)
    prgm = drop_reorders(prgm)
    prgm = flatten_plans(prgm)
    return normalize_names(prgm, bindings)


class LogicStandardizer(LogicLoader):
    """
    The LogicStandardizer applies a series of transformations to standardize
    logic statements into a canonical form. Any Logic is accepted as input, and
    the output logic should be a plan with only two forms of queries:
    1. Queries that perform a Reorder of a single argument.
    2. Queries that perform an Aggregate over a Reorder of a series of map-joins.
    3. Queries that perform a Reorder of a map-join between query lhs and (1)/(2).
    """

    def __init__(self, ctx: LogicLoader | None = None):
        if ctx is None:
            ctx = MockLogicLoader()
        self.ctx: LogicLoader = ctx

    def __call__(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, "TensorStats"],
        stats_factory: StatsFactory,
    ):
        prgm, bindings = standardize(prgm, bindings)
        return self.ctx(
            prgm,
            bindings,
            stats=stats,
            stats_factory=stats_factory,
        )
