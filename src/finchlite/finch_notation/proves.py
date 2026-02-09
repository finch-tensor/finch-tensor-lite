from collections.abc import Callable, Sequence
from functools import partial
from operator import add, eq, ge, le

from ..algebra import is_associative, is_idempotent, register_property
from ..algebra.utils import all_unique, intersect, is_disjoint, setdiff
from ..symbolic import Chain, Fixpoint, Memo, PreWalk, Rewrite
from .nodes import Cached, Call, NotationNode
from .nodes import Literal as L

# Register max and min as associative and commutative
for op in (max, min):
    register_property(op, "__call__", "is_associative", lambda op: True)
    register_property(op, "__call__", "is_commutative", lambda op: True)


def _find_first_call(args: Sequence[NotationNode], op: Callable):
    for i, arg in enumerate(args):
        if isinstance(arg, Call) and arg.op == L(op):
            return args[:i], arg.args, args[i + 1 :]
    return None


def rule_all_literals(ex):
    match ex:
        case Call(L(op), args):
            lit_args = []
            for arg in args:
                match arg:
                    case L(val):
                        lit_args.append(val)
                    case _:
                        return None
            return L(op(*lit_args))


def rule_idempotent_unique(ex):
    match ex:
        case Call(L(op), args) if is_idempotent(op) and not all_unique(args):
            return Call(L(op), tuple(set(args)))


def rule_associative_flatten(ex):
    match ex:
        case Call(L(op), args) if (
            is_associative(op) and (found := _find_first_call(args, op)) is not None
        ):
            before, call_args, after = found
            return Call(L(op), (*before, *call_args, *after))


def rule_equal_same(ex):
    match ex:
        case Call(L(op), (a, b)) if op == eq and a == b:
            return L(True)


def rule_ge(ex):
    match ex:
        case Call(L(op), (a, b)) if op == ge:
            return Call(L(eq), (a, Call(L(max), (a, b))))


def rule_le(ex):
    match ex:
        case Call(L(op), (a, b)) if op == le:
            return Call(L(eq), (Call(L(max), (a, b)), b))


# call(+, ~a..., call(max, ~b...), ~c...) =>
#   call(max, map(x -> call(+, a..., x, c...), b)...))
def rule_add_with(ex, func):
    match ex:
        case Call(L(op), args) if (
            op == add and (found := _find_first_call(args, func)) is not None
        ):
            before, call_args, after = found
            return Call(
                L(func), tuple(Call(L(add), (*before, ca, *after)) for ca in call_args)
            )


rule_add_with_max = partial(rule_add_with, func=max)
rule_add_with_min = partial(rule_add_with, func=min)


def rule_disjoint_nested(ex, func1, func2):
    match ex:
        case Call(L(op), args) if (
            op == func1 and (found := _find_first_call(args, func2)) is not None
        ):
            before, call_args, after = found
            if (found2 := _find_first_call(call_args, func1)) is not None:
                before2, call_args2, after2 = found2
                if not is_disjoint(call_args2, before) or not is_disjoint(
                    call_args2, after
                ):
                    return Call(
                        L(func1),
                        (
                            *before,
                            Call(
                                L(func2),
                                (
                                    *before2,
                                    *setdiff(call_args2, before + after),
                                    *after2,
                                ),
                            ),
                            *after,
                        ),
                    )
            return None


rule_disjoint_nested_max_min = partial(rule_disjoint_nested, func1=max, func2=min)
rule_disjoint_nested_min_max = partial(rule_disjoint_nested, func1=min, func2=max)


def rule_disjoint_flat_single(ex, func1, func2):
    match ex:
        case Call(L(op), args) if (
            op == func1 and (found := _find_first_call(args, func2)) is not None
        ):
            before, call_args, after = found
            if not is_disjoint(before, call_args) or not is_disjoint(call_args, after):
                return Call(L(func1), (*before, *after))
            return None


rule_disjoint_flat_single_max_min = partial(
    rule_disjoint_flat_single, func1=max, func2=min
)
rule_disjoint_flat_single_min_max = partial(
    rule_disjoint_flat_single, func1=min, func2=max
)


def rule_disjoint_flat_pair(ex, func1, func2):
    match ex:
        case Call(L(op), args) if (
            op == func1 and (found := _find_first_call(args, func2)) is not None
        ):
            before, call_args, after = found
            if (found2 := _find_first_call(after, func2)) is not None:
                before2, call_args2, after2 = found2
                if not is_disjoint(call_args, call_args2):
                    intersection = intersect(call_args, call_args2)
                    diff1 = setdiff(call_args, call_args2)
                    diff2 = setdiff(call_args2, call_args)
                    arg = Call(
                        L(func2),
                        (
                            *intersection,
                            Call(
                                L(func1), (Call(L(func2), diff1), Call(L(func2), diff2))
                            ),
                        ),
                    )
                    return Call(L(func1), (*before, arg, *before2, *after2))
            return None


rule_disjoint_flat_pair_max_min = partial(rule_disjoint_flat_pair, func1=max, func2=min)
rule_disjoint_flat_pair_min_max = partial(rule_disjoint_flat_pair, func1=min, func2=max)


def prove_rules():
    return [
        rule_all_literals,
        rule_idempotent_unique,
        rule_associative_flatten,
        rule_equal_same,
        rule_ge,
        rule_le,
        rule_add_with_max,
        rule_add_with_min,
        rule_disjoint_nested_max_min,
        rule_disjoint_nested_min_max,
        rule_disjoint_flat_single_max_min,
        rule_disjoint_flat_single_min_max,
        rule_disjoint_flat_pair_max_min,
        rule_disjoint_flat_pair_min_max,
    ]


def prove(root: NotationNode):
    def rule_extract_cached(ex):
        match ex:
            case Cached(_, L(val)):
                return val

    root = Rewrite(PreWalk(rule_extract_cached))(root)

    prove_cache: dict = {}

    # TODO: rename rewrite

    res = Rewrite(Fixpoint(PreWalk(Memo(Fixpoint(Chain(prove_rules())), prove_cache))))(
        root
    )
    match res:
        case L(val):
            return val
        case _:
            return False
