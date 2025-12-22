from functools import reduce

from finchlite.algebra.tensor import TensorFType

from ..algebra import overwrite
from ..finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicLoader,
    LogicStatement,
    MapJoin,
    MockLogicLoader,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    Table,
)
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
            case Query(lhs, rhs):
                rhs = Rewrite(PostWalk(rule_1))(rhs)
                return Plan((*stack, Query(lhs, rhs)))
            case Produces(args):
                args = tuple(Rewrite(PostWalk(rule_1))(arg) for arg in args)
                return Plan((*stack, Produces(args)))
            case _:
                return None

    return Rewrite(PostWalk(transform))(root)


def split_increments(root: LogicStatement) -> LogicStatement:
    def rule_2(stmt):
        match stmt:
            case Query(lhs, rhs):
                if lhs in PostOrderDFS(rhs):
                    var = Alias(gensym("A"))
                    new_query = Query(var, rhs)
                    new_root = Query(lhs, var)
                    return Plan((new_query, new_root))
        return None

    return Rewrite(PostWalk(rule_2))(root)


def standardize_query_roots(root: LogicStatement, bindings) -> LogicStatement:
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
            case Query(lhs, rhs):
                return Query(
                    lhs,
                    Aggregate(
                        Literal(overwrite),
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

    def rule_1(ex):
        match ex:
            case Query(lhs, rhs) as q if lhs in needed_swizzles:
                ndims = len(next(iter(needed_swizzles[lhs].items()))[0])
                idxs = tuple([Field(f"idx_{i}") for i in range(ndims)])
                swizzle_queries = tuple(
                    Query(
                        alias, Reorder(Table(lhs, idxs), tuple(idxs[p] for p in perm))
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


def push_fields(root: LogicStatement, bindings):

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
                relabeled_idxs = tuple(
                    reidx_dict.get(idx, idx) for idx in arg.fields()
                )
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
                isinstance(arg, Reorder) for arg in args
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

    return Rewrite(PostWalk(rule_2))(root)


def drop_with_aggregation(root: LogicStatement, bindings) -> LogicStatement:

    def rule_2(stmt):
        match stmt:
            case Query(lhs, Aggregate(op, init, Reorder(arg, idxs_1), idxs_2)):
                idxs_3 = tuple([idx for idx in arg.fields() if idx not in idxs_1])
                return Query(
                    lhs,
                    Aggregate(op, init, Reorder(arg, idxs_1 + idxs_3), idxs_2 + idxs_3),
                )

    return Rewrite(PostWalk(rule_2))(root)


class LogicStandardizer(LogicLoader):
    """
    The LogicStandardizer applies a series of transformations to standardize
    logic statements into a canonical form. Any Logic is accepted as input, and
    the output logic should be a plan with only two forms of queries:
    1. Queries that perform a Reorder of a single argument
    2. Queries that perform an Aggregate over a Reorder of a series of map-joins.
    """

    def __init__(self, ctx=None):
        if ctx is None:
            ctx = MockLogicLoader()
        self.ctx = ctx

    def __call__(self, prgm: LogicStatement, bindings):
        prgm = isolate_aggregates(prgm)
        prgm = split_increments(prgm)
        prgm = standardize_query_roots(prgm, bindings)
        prgm = push_fields(prgm, bindings)
        prgm = drop_reorders(prgm)
        prgm = drop_with_aggregation(prgm, bindings)
        prgm = concordize(prgm, bindings)
        prgm = drop_reorders(prgm)
        return self.ctx(prgm, bindings)
