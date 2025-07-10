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


def conjugate(x):
    """
    Computes the complex conjugate of the input number

    Parameters
    ----------
    x: Any
        The input number to compute the complex conjugate of.

    Returns
    ----------
    Any
        The complex conjugate of the input number. If the input is not a complex number,
        it returns the input unchanged.
    """
    if hasattr(x, "conjugate"):
        return x.conjugate()
    return x


# register the conjugate operation return type. The conjugate operation
# preserves the element type of the input tensor.
algebra.register_property(
    conjugate,
    "__call__",
    "return_type",
    lambda obj, x: x,
)

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


class InitWrite:
    """
    InitWrite may assert that its first argument is
    equal to z, and returns its second argument. This is useful when you want to
    communicate to the compiler that the tensor has already been initialized to
    a specific value.
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, x, y):
        assert x == self.value, f"Expected {self.value}, got {x}"
        return y


algebra.register_property(
    InitWrite,
    "__call__",
    "return_type",
    lambda op, x, y: y,
)


def overwrite(x, y):
    """
    overwrite(x, y) returns y always.
    """
    return y


algebra.register_property(
    overwrite,
    "__call__",
    "return_type",
    lambda op, x, y: y,
)


def first_arg(*args):
    """
    Returns the first argument passed to it.
    """
    return args[0] if args else None


algebra.register_property(
    first_arg,
    "__call__",
    "return_type",
    # args[0] is the function name
    lambda *args: args[1],
)


def identity(x):
    """
    Returns the input value unchanged.
    """
    return x


algebra.register_property(
    identity,
    "__call__",
    "return_type",
    lambda op, x: x,
)
