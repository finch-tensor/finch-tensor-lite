# functions ported from ops.py
import operator
from functools import reduce
from typing import Any

import numpy as np

from .algebra import (
    COperator,
    FinchOperator,
    NumbaOperator,
    promote_type,
    type_max,
    type_min,
)


class CNAryOperator(COperator):
    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        assert len(args) > 0

        if len(args) == 1:
            return f"{self.c_symbol}{ctx(args[0])}"
        return f" {self.c_symbol} ".join(map(ctx, args))


class CBinaryOperator(COperator):
    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        a, b = args
        return f"{ctx(a)} {self.c_symbol} {ctx(b)}"


class CNUnaryOperator(COperator):
    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        return f"{self.c_symbol}{ctx(args[0])}"


class ReflexiveFinchOperator(FinchOperator):
    def return_type(self, a: Any, b: Any) -> type:
        return type(self(a(True), b(True)))


class UnaryFinchOperator(FinchOperator):
    def return_type(self, a: Any) -> type:
        return type(self(a(True)))


class ComparisonFinchOperator(FinchOperator):
    def return_type(self, a: Any, b: Any) -> type:
        return bool


class Add(ReflexiveFinchOperator, CNAryOperator, NumbaOperator):
    is_associative = True
    is_commutative = True

    @property
    def c_symbol(self) -> str:
        return "+"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.add, args)

    def is_identity(self, arg: Any) -> bool:
        return arg == 0

    def is_annihilator(self, arg: Any) -> bool:
        return np.isinf(arg)

    def repeat_operator(self):
        return mul

    def init_value(self, type_: type) -> Any:
        return type_(0)

    def numba_name(self) -> str:
        return "+"


add = Add()


class Mul(ReflexiveFinchOperator, CNAryOperator, NumbaOperator):
    is_associative = True
    is_commutative = True

    @property
    def c_symbol(self) -> str:
        return "*"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.mul, args)

    def is_identity(self, arg: Any) -> bool:
        return arg == 1

    def repeat_operator(self):
        return pow

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, (Add, Sub))

    def is_annihilator(self, val):
        return val == 0

    def init_value(self, type_: type) -> Any:
        return type_(1)

    def numba_name(self) -> str:
        return "*"


mul = Mul()


class Sub(ReflexiveFinchOperator, CBinaryOperator, NumbaOperator):
    @property
    def c_symbol(self) -> str:
        return "-"

    def __call__(self, a: Any, b: Any):
        return operator.sub(a, b)

    def numba_name(self) -> str:
        return "-"


sub = Sub()


class MatMul(ReflexiveFinchOperator):
    is_associative = True

    def __call__(self, a: Any, b: Any):
        return operator.matmul(a, b)


matmul = MatMul()


class TrueDiv(ReflexiveFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "/"

    def __call__(self, a: Any, b: Any):
        return operator.truediv(a, b)

    def is_identity(self, arg):
        return arg == 1


truediv = TrueDiv()


class FloorDiv(ReflexiveFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "/"

    def __call__(self, a: Any, b: Any):
        return operator.floordiv(a, b)


floordiv = FloorDiv()


class Mod(ReflexiveFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "%"

    def __call__(self, a: Any, b: Any):
        return operator.mod(a, b)


mod = Mod()


class DivMod(ReflexiveFinchOperator):
    def __call__(self, a: Any, b: Any):
        return divmod(a, b)


divmod = DivMod()


class Pow(ReflexiveFinchOperator, COperator):
    @property
    def c_symbol(self) -> str:
        return "pow"

    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        a, b = args
        return f"pow({ctx(a)}, {ctx(b)})"

    def __call__(self, a: Any, b: Any):
        return operator.pow(a, b)

    def is_identity(self, arg):
        return arg == 1

    def is_annihilator(self, arg):
        return arg == 0


pow = Pow()


class LShift(ReflexiveFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "<<"

    def __call__(self, a: Any, b: Any):
        return operator.lshift(a, b)

    def is_identity(self, arg):
        return arg == 0


lshift = LShift()


class RShift(ReflexiveFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return ">>"

    def __call__(self, a: Any, b: Any):
        return operator.rshift(a, b)

    def is_identity(self, arg):
        return arg == 0


rshift = RShift()


class And(ReflexiveFinchOperator, CNAryOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    @property
    def c_symbol(self) -> str:
        return "&"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.and_, args)

    def is_identity(self, arg):
        return bool(arg)

    def is_annihilator(self, arg):
        return not bool(arg)

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, (Or, Xor))

    def init_value(self, type_: type) -> Any:
        return type_(True)


and_ = And()


class Xor(ReflexiveFinchOperator, CNAryOperator):
    is_associative = True
    is_commutative = True

    @property
    def c_symbol(self) -> str:
        return "^"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.xor, args)

    def is_identity(self, arg):
        return arg == 0

    def init_value(self, type_: type) -> Any:
        return type_(False)


xor = Xor()


class Or(ReflexiveFinchOperator, CNAryOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    @property
    def c_symbol(self) -> str:
        return "|"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.or_, args)

    def is_identity(self, arg):
        return not bool(arg)

    def is_annihilator(self, arg):
        return bool(arg)

    def is_distributive(self, other_op: "FinchOperator") -> bool:
        return isinstance(other_op, And)

    def init_value(self, type_: type) -> Any:
        return type_(False)


or_ = Or()


class Not(CNUnaryOperator):
    @property
    def c_symbol(self) -> str:
        return "!"


not_ = Not()


class Abs(UnaryFinchOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return operator.abs(a)


abs = Abs()


class Pos(UnaryFinchOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return operator.pos(a)


pos = Pos()


class Neg(UnaryFinchOperator):
    def __call__(self, a: Any):
        return operator.neg(a)


neg = Neg()


class Invert(UnaryFinchOperator, CNUnaryOperator):
    @property
    def c_symbol(self) -> str:
        return "~"

    def __call__(self, a: Any):
        return operator.invert(a)


invert = Invert()


class Eq(ComparisonFinchOperator, CBinaryOperator, NumbaOperator):
    @property
    def c_symbol(self) -> str:
        return "=="

    def numba_name(self) -> str:
        return "=="

    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.eq(a, b)


eq = Eq()


class Ne(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "!="

    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.ne(a, b)


ne = Ne()


class Gt(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return ">"

    def __call__(self, a: Any, b: Any):
        return operator.gt(a, b)


gt = Gt()


class Lt(ComparisonFinchOperator, CBinaryOperator, NumbaOperator):
    @property
    def c_symbol(self) -> str:
        return "<"

    def numba_name(self) -> str:
        return "<"

    def __call__(self, a: Any, b: Any):
        return operator.lt(a, b)


lt = Lt()


class Ge(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return ">="

    def __call__(self, a: Any, b: Any):
        return operator.ge(a, b)


ge = Ge()


class Le(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "<="

    def __call__(self, a: Any, b: Any):
        return operator.le(a, b)


le = Le()


class BinaryFloatOperator(FinchOperator):
    def return_type(self, a: Any, b: Any) -> type:
        return float


class UnaryOperator(FinchOperator):
    def return_type(self, a: Any) -> type:
        # TODO: Temporary implementation
        if a is np.float16:
            return a
        if a is np.float32:
            return a
        if np.can_cast(a, np.float64):
            return np.float64
        if a is np.complex64:
            return a
        if a is np.complex128:
            return a
        raise TypeError(f"Unsupported operand type for {self}: {a}")


class UnaryBoolOperator(FinchOperator):
    def return_type(self, a: Any) -> type:
        return bool


class BinaryBoolOperator(FinchOperator):
    def return_type(self, a: Any, b: Any) -> type:
        return bool


class LogicalBinaryOperator(BinaryBoolOperator):
    is_associative = True
    is_commutative = True


class Divide(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.divide(a, b)

    def is_identity(self, val) -> bool:
        return val == 1


divide = Divide()


class LogAddExp(BinaryFloatOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = False

    def __call__(self, a, b):
        return np.logaddexp(a, b)

    def is_identity(self, val) -> bool:
        return val == -np.inf

    def is_annihilator(self, val) -> bool:
        return val == np.inf

    def init_value(self, type_: type) -> Any:
        return -np.inf


logaddexp = LogAddExp()


class LogicalAnd(LogicalBinaryOperator):
    is_idempotent = True

    def __call__(self, a, b):
        return np.logical_and(a, b)

    def is_identity(self, val) -> bool:
        return bool(val)

    def is_annihilator(self, val) -> bool:
        return not bool(val)

    def is_distributive(self, other_op: FinchOperator) -> bool:
        return isinstance(other_op, (LogicalOr, LogicalXor))

    def init_value(self, type_: type) -> Any:
        return True


logical_and = LogicalAnd()


class LogicalOr(LogicalBinaryOperator):
    is_idempotent = True

    def __call__(self, a, b):
        return np.logical_or(a, b)

    def is_identity(self, val) -> bool:
        return not bool(val)

    def is_annihilator(self, val) -> bool:
        return bool(val)

    def is_distributive(self, other_op: FinchOperator) -> bool:
        return isinstance(other_op, LogicalAnd)

    def init_value(self, type_: type) -> Any:
        return False


logical_or = LogicalOr()


class LogicalXor(LogicalBinaryOperator):
    is_idempotent = False

    def __call__(self, a, b):
        return np.logical_xor(a, b)

    def is_identity(self, val) -> bool:
        return not bool(val)

    def init_value(self, type_: type) -> Any:
        return False


logical_xor = LogicalXor()


class LogicalNot(UnaryBoolOperator):
    def __call__(self, a):
        return np.logical_not(a)


logical_not = LogicalNot()


class Truth(UnaryBoolOperator):
    def __call__(self, a: Any):
        return bool(a)


truth = Truth()


class Min(FinchOperator, NumbaOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return min(a, b)

    def return_type(self, a: Any, b: Any) -> type:
        return type(min(a(True), b(True)))

    def is_identity(self, val) -> bool:
        return val == np.inf

    def init_value(self, type_: type):
        return type_max(type_)

    def numba_name(self) -> str:
        return "min"

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return f"min({', '.join(map(ctx, args))})"


min = Min()


class Max(FinchOperator, NumbaOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return max(a, b)

    def return_type(self, a: Any, b: Any) -> type:
        return type(max(a(True), b(True)))

    def is_identity(self, val) -> bool:
        return val == -np.inf

    def init_value(self, type_: type):
        return type_min(type_)

    def numba_name(self) -> str:
        return "max"

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return f"max({', '.join(map(ctx, args))})"


max = Max()


class Remainder(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.remainder(a, b)


remainder = Remainder()


class Hypot(BinaryFloatOperator):
    is_commutative = True

    def __call__(self, a, b):
        return np.hypot(a, b)


hypot = Hypot()


class Atan2(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.atan2(a, b)


atan2 = Atan2()


class Copysign(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.copysign(a, b)


copysign = Copysign()


class Nextafter(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.nextafter(a, b)


nextafter = Nextafter()


class IsFinite(UnaryBoolOperator):
    def __call__(self, a):
        return np.isfinite(a)


isfinite = IsFinite()


class IsInf(UnaryBoolOperator):
    def __call__(self, a):
        return np.isinf(a)


isinf = IsInf()


class IsNan(UnaryBoolOperator):
    def __call__(self, a):
        return np.isnan(a)


isnan = IsNan()


class Real(UnaryOperator):
    def __call__(self, a):
        return np.real(a)

    def return_type(self, a: Any) -> type:
        return float


real = Real()


class Imag(UnaryOperator):
    def __call__(self, a: Any):
        return np.imag(a)

    def return_type(self, a: Any) -> type:
        return float


imag = Imag()


class Clip(FinchOperator):
    def __call__(self, a: Any, b: Any, c: Any):
        return np.clip(a, b, c)

    def return_type(self, a: Any, b: Any, c: Any) -> type:
        return float


clip = Clip()


class Equal(BinaryBoolOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.equal(a, b)


equal = Equal()


class NotEqual(BinaryBoolOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.not_equal(a, b)


not_equal = NotEqual()


class Less(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.less(a, b)


less = Less()


class LessEqual(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.less_equal(a, b)


less_equal = LessEqual()


class Greater(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater(a, b)


greater = Greater()


class GreaterEqual(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater_equal(a, b)


greater_equal = GreaterEqual()


class Reciprocal(UnaryOperator):
    def __call__(self, a: Any):
        return np.reciprocal(a)


reciprocal = Reciprocal()


class Sin(UnaryOperator):
    def __call__(self, a: Any):
        return np.sin(a)


sin = Sin()


class Cos(UnaryOperator):
    def __call__(self, a: Any):
        return np.cos(a)


cos = Cos()


class Tan(UnaryOperator):
    def __call__(self, a: Any):
        return np.tan(a)


tan = Tan()


class Sinh(UnaryOperator):
    def __call__(self, a: Any):
        return np.sinh(a)


sinh = Sinh()


class Cosh(UnaryOperator):
    def __call__(self, a: Any):
        return np.cosh(a)


cosh = Cosh()


class Tanh(UnaryOperator):
    def __call__(self, a: Any):
        return np.tanh(a)


tanh = Tanh()


class Atan(UnaryOperator):
    def __call__(self, a: Any):
        return np.atan(a)


atan = Atan()


class Asinh(UnaryOperator):
    def __call__(self, a: Any):
        return np.asinh(a)


asinh = Asinh()


class Asin(UnaryOperator):
    def __call__(self, a: Any):
        return np.asin(a)


asin = Asin()


class Acos(UnaryOperator):
    def __call__(self, a: Any):
        return np.acos(a)


acos = Acos()


class Acosh(UnaryOperator):
    def __call__(self, a: Any):
        return np.acosh(a)


acosh = Acosh()


class Atanh(UnaryOperator):
    def __call__(self, a: Any):
        return np.atanh(a)


atanh = Atanh()


class Round(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.round(a)


round = Round()


class Floor(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.floor(a)


floor = Floor()


class Ceil(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.ceil(a)


ceil = Ceil()


class Trunc(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.trunc(a)


trunc = Trunc()


class Exp(UnaryOperator):
    def __call__(self, a: Any):
        return np.exp(a)


exp = Exp()


class Expm1(UnaryOperator):
    def __call__(self, a: Any):
        return np.expm1(a)


expm1 = Expm1()


class Log(UnaryOperator):
    def __call__(self, a: Any):
        return np.log(a)


log = Log()


class Log1p(UnaryOperator):
    def __call__(self, a: Any):
        return np.log1p(a)


log1p = Log1p()


class Log2(UnaryOperator):
    def __call__(self, a: Any):
        return np.log2(a)


log2 = Log2()


class Log10(UnaryOperator):
    def __call__(self, a: Any):
        return np.log10(a)


log10 = Log10()


class Signbit(UnaryBoolOperator):
    def __call__(self, a: Any):
        return np.signbit(a)


signbit = Signbit()


class Sqrt(UnaryOperator):
    def __call__(self, a: Any):
        return np.sqrt(a)


sqrt = Sqrt()


class Square(UnaryOperator):
    def __call__(self, a: Any):
        return np.square(a)


square = Square()


class Sign(UnaryOperator):
    def __call__(self, a: Any):
        return np.sign(a)


sign = Sign()


class PromoteMin(FinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a: Any, b: Any):
        cast = promote_type(type(a), type(b))
        return cast(min(a, b))

    def return_type(self, a: Any, b: Any) -> type:
        return promote_type(a, b)

    def init_value(self, arg):
        return type_max(arg)


promote_min = PromoteMin()


class PromoteMax(FinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a: Any, b: Any):
        cast = promote_type(type(a), type(b))
        return max(cast(a), cast(b))

    def return_type(self, a: Any, b: Any) -> type:
        return promote_type(a, b)

    def init_value(self, arg):
        return type_min(arg)


promote_max = PromoteMax()


class InitWrite(FinchOperator, NumbaOperator):
    """
    InitWrite may assert that its first argument is
    equal to z, and returns its second argument. This is useful when you want to
    communicate to the compiler that the tensor has already been initialized to
    a specific value.
    """

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, InitWrite) and self.value == other.value

    def __hash__(self):
        return hash((self.value,))

    def __call__(self, x: Any, y: Any):
        assert x == self.value, f"Expected {self.value}, got {x}"
        return y

    def return_type(self, x: Any, y: Any) -> type:
        return y

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return ctx(args[1])


def init_write(value):
    return InitWrite(value)


class Overwrite(FinchOperator):
    """
    Overwrite(x, y) returns y always.
    """

    def __call__(self, x: Any, y: Any):
        return y

    def return_type(self, x: Any, y: Any) -> type:
        return y


overwrite = Overwrite()


class FirstArg(FinchOperator):
    """
    Returns the first argument passed to it.
    """

    def __call__(self, *args):
        return args[0] if args else None

    def return_type(self, *args) -> type:
        return args[0]


first_arg = FirstArg()


class Identity(FinchOperator):
    """
    Returns the input value unchanged.
    """

    is_idempotent = True

    def __call__(self, x: Any):
        return x

    def return_type(self, x: Any) -> type:
        return x


identity = Identity()


class Conjugate(FinchOperator):
    """
    Returns the complex conjugate of the input value.
    """

    def __call__(self, x: Any):
        return np.conjugate(x)

    def return_type(self, x: Any) -> type:
        return x


conjugate = Conjugate()


class MakeTuple(FinchOperator, NumbaOperator):
    is_commutative = False
    is_associative = False

    def __call__(self, *args: Any) -> tuple:
        return tuple(args)

    def return_type(self, *args: Any) -> Any:
        from finchlite.finch_assembly.struct import TupleFType

        return TupleFType.from_tuple(args)

    def numba_literal(self, val: Any, ctx: Any, *args: Any):
        return f"({','.join([ctx(arg) for arg in args])},)"


make_tuple = MakeTuple()


class Scansearch(FinchOperator, NumbaOperator):
    """
    Scansearch is a search operator that performs a scan search on a sorted array.

    It takes an array `arr`, a value `x`, and search bounds `lo` and `hi`, and returns
    the index of the smallest element in `arr` that is greater than or equal to `x`.
    If all elements in `arr` are less than `x`, it returns `hi`.
    """

    @staticmethod
    def _func(
        arr: np.ndarray, x: np.integer, lo: np.integer, hi: np.integer
    ) -> np.integer:
        dtype = np.array(lo).dtype.type
        u = dtype(1)
        d = dtype(1)
        p = lo

        # searching for binary search bounds
        while p < hi and arr[p] < x:
            d <<= 0x01
            p += d
        lo = p - d
        hi = min(p, hi) + u  # type: ignore[call-overload]

        # binary searching within those bounds
        while lo < hi - u:
            m = lo + ((hi - lo) >> 0x01)
            if arr[m] < x:
                lo = m
            else:
                hi = m

        return hi

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def return_type(self, arr, x, lo, hi) -> type:
        return hi

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        arr = args[0]
        x = args[1]
        lo = args[2]
        hi = args[3]
        return f"scansearch({ctx(arr)}, {ctx(x)}, {ctx(lo)}, {ctx(hi)})"


scansearch = Scansearch()

__all__ = [
    "abs",
    "abs",
    "acos",
    "acosh",
    "add",
    "and_",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "clip",
    "conjugate",
    "copysign",
    "cos",
    "cosh",
    "divide",
    "divmod",
    "eq",
    "equal",
    "exp",
    "expm1",
    "first_arg",
    "floor",
    "floordiv",
    "ge",
    "greater",
    "greater_equal",
    "gt",
    "hypot",
    "identity",
    "imag",
    "invert",
    "isfinite",
    "isinf",
    "isnan",
    "le",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "lshift",
    "lt",
    "make_tuple",
    "matmul",
    "max",
    "min",
    "mod",
    "mul",
    "ne",
    "neg",
    "nextafter",
    "not_equal",
    "or_",
    "overwrite",
    "pos",
    "pow",
    "promote_max",
    "promote_min",
    "real",
    "reciprocal",
    "remainder",
    "round",
    "rshift",
    "scansearch",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "sub",
    "tan",
    "tanh",
    "truediv",
    "trunc",
    "truth",
    "xor",
]
