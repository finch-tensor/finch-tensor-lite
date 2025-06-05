from . import algebra


def and_test(a, b):
    return a & b


def or_test(a, b):
    return a | b


def not_test(a):
    return not a


def ifelse(a, b, c):
    return a if c else b


def promote_min(a, b):
    cast = algebra.promote_type(a, b)
    return cast(min(a, b))


def promote_max(a, b):
    cast = algebra.promote_type(a, b)
    return max(cast(a), cast(b))


algebra.register_property(
    promote_min,
    "__call__",
    "return_type",
    lambda op, a, b: algebra.promote_type(a, b),
)


algebra.register_property(
    promote_max,
    "__call__",
    "return_type",
    lambda op, a, b: algebra.promote_type(a, b),
)

algebra.register_property(
    promote_min, "__call__", "init_value", lambda op, arg: algebra.type_max(arg)
)
algebra.register_property(
    promote_max, "__call__", "init_value", lambda op, arg: algebra.type_min(arg)
)
