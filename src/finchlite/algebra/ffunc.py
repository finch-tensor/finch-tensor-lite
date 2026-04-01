# AI modified: 2026-04-01T17:18:51Z 0de216cc18e91710a9b1a0328f5b181137d8901b
# AI modified: 2026-04-01T17:28:42Z 0de216cc18e91710a9b1a0328f5b181137d8901b
# AI modified: 2026-04-01T17:34:47Z d369513eef4124a0bcb300a625b553c445a8a73e
# AI modified: 2026-04-01T18:10:00Z 09431a5eedd67043b459fb6eafbfc5fd936fcf19
# AI modified: 2026-04-01T18:25:00Z 09431a5eedd67043b459fb6eafbfc5fd936fcf19
# functions ported from ops.py
import builtins
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

    def __repr__(self) -> str:
        return "finchlite.ffunc.add"

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

    def __repr__(self) -> str:
        return "finchlite.ffunc.mul"

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
    def __repr__(self) -> str:
        return "finchlite.ffunc.sub"
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

    def __repr__(self) -> str:
        return "finchlite.ffunc.matmul"

    def __call__(self, a: Any, b: Any):
        return operator.matmul(a, b)


matmul = MatMul()


class TrueDiv(ReflexiveFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "/"

    def __repr__(self) -> str:
        return "finchlite.ffunc.truediv"

    def __call__(self, a: Any, b: Any):
        return operator.truediv(a, b)

    def is_identity(self, arg):
        return arg == 1


truediv = TrueDiv()


class FloorDiv(ReflexiveFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "/"

    def __repr__(self) -> str:
        return "finchlite.ffunc.floordiv"

    def __call__(self, a: Any, b: Any):
        return operator.floordiv(a, b)


floordiv = FloorDiv()


class Mod(ReflexiveFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "%"

    def __repr__(self) -> str:
        return "finchlite.ffunc.mod"

    def __call__(self, a: Any, b: Any):
        return operator.mod(a, b)


mod = Mod()


class DivMod(ReflexiveFinchOperator):
    def __call__(self, a: Any, b: Any):
        return divmod(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.divmod"


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

    def __repr__(self) -> str:
        return "finchlite.ffunc.pow"


pow = Pow()


class LShift(ReflexiveFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "<<"

    def __call__(self, a: Any, b: Any):
        return operator.lshift(a, b)

    def is_identity(self, arg):
        return arg == 0

    def __repr__(self) -> str:
        return "finchlite.ffunc.lshift"


lshift = LShift()


class RShift(ReflexiveFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return ">>"

    def __call__(self, a: Any, b: Any):
        return operator.rshift(a, b)

    def is_identity(self, arg):
        return arg == 0

    def __repr__(self) -> str:
        return "finchlite.ffunc.rshift"


rshift = RShift()


class And(ReflexiveFinchOperator, CNAryOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __repr__(self) -> str:
        return "finchlite.ffunc.and_"

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

    def __repr__(self) -> str:
        return "finchlite.ffunc.xor"

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

    def __repr__(self) -> str:
        return "finchlite.ffunc.or_"

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

    def __repr__(self) -> str:
        return "finchlite.ffunc.not_"


not_ = Not()


class Abs(UnaryFinchOperator):
    is_idempotent = True

    def __repr__(self) -> str:
        return "finchlite.ffunc.abs"

    def __call__(self, a: Any):
        return operator.abs(a)


abs = Abs()


class Pos(UnaryFinchOperator):
    is_idempotent = True

    def __repr__(self) -> str:
        return "finchlite.ffunc.pos"

    def __call__(self, a: Any):
        return operator.pos(a)


pos = Pos()


class Neg(UnaryFinchOperator):
    def __call__(self, a: Any):
        return operator.neg(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.neg"


neg = Neg()


class Invert(UnaryFinchOperator, CNUnaryOperator):
    @property
    def c_symbol(self) -> str:
        return "~"

    def __call__(self, a: Any):
        return operator.invert(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.invert"


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

    def __repr__(self) -> str:
        return "finchlite.ffunc.eq"


eq = Eq()


class Ne(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "!="

    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.ne(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.ne"


ne = Ne()


class Gt(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return ">"

    def __call__(self, a: Any, b: Any):
        return operator.gt(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.gt"


gt = Gt()


class Lt(ComparisonFinchOperator, CBinaryOperator, NumbaOperator):
    @property
    def c_symbol(self) -> str:
        return "<"

    def numba_name(self) -> str:
        return "<"

    def __call__(self, a: Any, b: Any):
        return operator.lt(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.lt"


lt = Lt()


class Ge(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return ">="

    def __call__(self, a: Any, b: Any):
        return operator.ge(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.ge"


ge = Ge()


class Le(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "<="

    def __call__(self, a: Any, b: Any):
        return operator.le(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.le"


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

    def __repr__(self) -> str:
        return "finchlite.ffunc.divide"


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

    def __repr__(self) -> str:
        return "finchlite.ffunc.logaddexp"


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

    def __repr__(self) -> str:
        return "finchlite.ffunc.logical_and"


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

    def __repr__(self) -> str:
        return "finchlite.ffunc.logical_or"


logical_or = LogicalOr()


class LogicalXor(LogicalBinaryOperator):
    is_idempotent = False

    def __call__(self, a, b):
        return np.logical_xor(a, b)

    def is_identity(self, val) -> bool:
        return not bool(val)

    def init_value(self, type_: type) -> Any:
        return False

    def __repr__(self) -> str:
        return "finchlite.ffunc.logical_xor"


logical_xor = LogicalXor()


class LogicalNot(UnaryBoolOperator):
    def __call__(self, a):
        return np.logical_not(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.logical_not"


logical_not = LogicalNot()


class Truth(UnaryBoolOperator):
    def __call__(self, a: Any):
        return bool(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.truth"


truth = Truth()


class Min(FinchOperator, NumbaOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return builtins.min(a, b)

    def return_type(self, a: Any, b: Any) -> type:
        return type(builtins.min(a(True), b(True)))

    def is_identity(self, val) -> bool:
        return val == np.inf

    def init_value(self, type_: type):
        return type_max(type_)

    def numba_name(self) -> str:
        return "min"

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return f"min({', '.join(map(ctx, args))})"

    def __repr__(self) -> str:
        return "finchlite.ffunc.min"


min = Min()


class Max(FinchOperator, NumbaOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return builtins.max(a, b)

    def return_type(self, a: Any, b: Any) -> type:
        return type(builtins.max(a(True), b(True)))

    def is_identity(self, val) -> bool:
        return val == -np.inf

    def init_value(self, type_: type):
        return type_min(type_)

    def numba_name(self) -> str:
        return "max"

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return f"max({', '.join(map(ctx, args))})"

    def __repr__(self) -> str:
        return "finchlite.ffunc.max"


max = Max()


class Remainder(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.remainder(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.remainder"


remainder = Remainder()


class Hypot(BinaryFloatOperator):
    is_commutative = True

    def __call__(self, a, b):
        return np.hypot(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.hypot"


hypot = Hypot()


class Atan2(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.atan2(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.atan2"


atan2 = Atan2()


class Copysign(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.copysign(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.copysign"


copysign = Copysign()


class Nextafter(BinaryFloatOperator):
    def __call__(self, a, b):
        return np.nextafter(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.nextafter"


nextafter = Nextafter()


class IsFinite(UnaryBoolOperator):
    def __call__(self, a):
        return np.isfinite(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.isfinite"


isfinite = IsFinite()


class IsInf(UnaryBoolOperator):
    def __call__(self, a):
        return np.isinf(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.isinf"


isinf = IsInf()


class IsNan(UnaryBoolOperator):
    def __call__(self, a):
        return np.isnan(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.isnan"


isnan = IsNan()


class Real(UnaryOperator):
    def __call__(self, a):
        return np.real(a)

    def return_type(self, a: Any) -> type:
        return float

    def __repr__(self) -> str:
        return "finchlite.ffunc.real"


real = Real()


class Imag(UnaryOperator):
    def __call__(self, a: Any):
        return np.imag(a)

    def return_type(self, a: Any) -> type:
        return float

    def __repr__(self) -> str:
        return "finchlite.ffunc.imag"


imag = Imag()


class Clip(FinchOperator):
    def __call__(self, a: Any, b: Any, c: Any):
        return np.clip(a, b, c)

    def return_type(self, a: Any, b: Any, c: Any) -> type:
        return float

    def __repr__(self) -> str:
        return "finchlite.ffunc.clip"


clip = Clip()


class Equal(BinaryBoolOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.equal(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.equal"


equal = Equal()


class NotEqual(BinaryBoolOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.not_equal(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.not_equal"


not_equal = NotEqual()


class Less(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.less(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.less"


less = Less()


class LessEqual(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.less_equal(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.less_equal"


less_equal = LessEqual()


class Greater(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.greater"


greater = Greater()


class GreaterEqual(BinaryBoolOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater_equal(a, b)

    def __repr__(self) -> str:
        return "finchlite.ffunc.greater_equal"


greater_equal = GreaterEqual()


class Reciprocal(UnaryOperator):
    def __call__(self, a: Any):
        return np.reciprocal(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.reciprocal"


reciprocal = Reciprocal()


class Sin(UnaryOperator):
    def __call__(self, a: Any):
        return np.sin(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.sin"


sin = Sin()


class Cos(UnaryOperator):
    def __call__(self, a: Any):
        return np.cos(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.cos"


cos = Cos()


class Tan(UnaryOperator):
    def __call__(self, a: Any):
        return np.tan(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.tan"


tan = Tan()


class Sinh(UnaryOperator):
    def __call__(self, a: Any):
        return np.sinh(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.sinh"


sinh = Sinh()


class Cosh(UnaryOperator):
    def __call__(self, a: Any):
        return np.cosh(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.cosh"


cosh = Cosh()


class Tanh(UnaryOperator):
    def __call__(self, a: Any):
        return np.tanh(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.tanh"


tanh = Tanh()


class Atan(UnaryOperator):
    def __call__(self, a: Any):
        return np.atan(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.atan"


atan = Atan()


class Asinh(UnaryOperator):
    def __call__(self, a: Any):
        return np.asinh(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.asinh"


asinh = Asinh()


class Asin(UnaryOperator):
    def __call__(self, a: Any):
        return np.asin(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.asin"


asin = Asin()


class Acos(UnaryOperator):
    def __call__(self, a: Any):
        return np.acos(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.acos"


acos = Acos()


class Acosh(UnaryOperator):
    def __call__(self, a: Any):
        return np.acosh(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.acosh"


acosh = Acosh()


class Atanh(UnaryOperator):
    def __call__(self, a: Any):
        return np.atanh(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.atanh"


atanh = Atanh()

# Backward-compatible aliases.
arcsin = asin
arccos = acos
arctan = atan
arcsinh = asinh
arccosh = acosh
arctanh = atanh


class Round(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.round(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.round"


round = Round()


class Floor(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.floor(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.floor"


floor = Floor()


class Ceil(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.ceil(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.ceil"


ceil = Ceil()


class Trunc(UnaryOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.trunc(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.trunc"


trunc = Trunc()


class Exp(UnaryOperator):
    def __call__(self, a: Any):
        return np.exp(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.exp"


exp = Exp()


class Expm1(UnaryOperator):
    def __call__(self, a: Any):
        return np.expm1(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.expm1"


expm1 = Expm1()


class Log(UnaryOperator):
    def __call__(self, a: Any):
        return np.log(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.log"


log = Log()


class Log1p(UnaryOperator):
    def __call__(self, a: Any):
        return np.log1p(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.log1p"


log1p = Log1p()


class Log2(UnaryOperator):
    def __call__(self, a: Any):
        return np.log2(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.log2"


log2 = Log2()


class Log10(UnaryOperator):
    def __call__(self, a: Any):
        return np.log10(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.log10"


log10 = Log10()


class Signbit(UnaryBoolOperator):
    def __call__(self, a: Any):
        return np.signbit(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.signbit"


signbit = Signbit()


class Sqrt(UnaryOperator):
    def __call__(self, a: Any):
        return np.sqrt(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.sqrt"


sqrt = Sqrt()


class Square(UnaryOperator):
    def __call__(self, a: Any):
        return np.square(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.square"


square = Square()


class Sign(UnaryOperator):
    def __call__(self, a: Any):
        return np.sign(a)

    def __repr__(self) -> str:
        return "finchlite.ffunc.sign"


sign = Sign()


class PromoteMin(FinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a: Any, b: Any):
        cast = promote_type(type(a), type(b))
        return cast(builtins.min(a, b))

    def return_type(self, a: Any, b: Any) -> type:
        return promote_type(a, b)

    def init_value(self, arg):
        return type_max(arg)

    def __repr__(self) -> str:
        return "finchlite.ffunc.promote_min"


promote_min = PromoteMin()


class PromoteMax(FinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a: Any, b: Any):
        cast = promote_type(type(a), type(b))
        return builtins.max(cast(a), cast(b))

    def return_type(self, a: Any, b: Any) -> type:
        return promote_type(a, b)

    def init_value(self, arg):
        return type_min(arg)

    def __repr__(self) -> str:
        return "finchlite.ffunc.promote_max"


promote_max = PromoteMax()


class _InitWrite(FinchOperator, NumbaOperator):
    """
    init_write may assert that its first argument is
    equal to z, and returns its second argument. This is useful when you want to
    communicate to the compiler that the tensor has already been initialized to
    a specific value.
    """

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, _InitWrite) and self.value == other.value

    def __hash__(self):
        return hash((self.value,))

    def __call__(self, x: Any, y: Any):
        assert x == self.value, f"Expected {self.value}, got {x}"
        return y

    def return_type(self, x: Any, y: Any) -> type:
        return y

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return ctx(args[1])

    def __repr__(self) -> str:
        return "finchlite.ffunc._initwrite"


def init_write(value):
    return _InitWrite(value)


class Overwrite(FinchOperator):
    """
    Overwrite(x, y) returns y always.
    """

    def __call__(self, x: Any, y: Any):
        return y

    def return_type(self, x: Any, y: Any) -> type:
        return y

    def __repr__(self) -> str:
        return "finchlite.ffunc.overwrite"


overwrite = Overwrite()


class FirstArg(FinchOperator):
    """
    Returns the first argument passed to it.
    """

    def __call__(self, *args):
        return args[0] if args else None

    def return_type(self, *args) -> type:
        return args[0]

    def __repr__(self) -> str:
        return "finchlite.ffunc.first_arg"


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

    def __repr__(self) -> str:
        return "finchlite.ffunc.identity"


identity = Identity()


class Conjugate(FinchOperator):
    """
    Returns the complex conjugate of the input value.
    """

    def __call__(self, x: Any):
        return np.conjugate(x)

    def return_type(self, x: Any) -> type:
        return x

    def __repr__(self) -> str:
        return "finchlite.ffunc.conjugate"


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

    def __repr__(self) -> str:
        return "finchlite.ffunc.make_tuple"


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

    def __repr__(self) -> str:
        return "finchlite.ffunc.scansearch"


scansearch = Scansearch()

__all__ = [
    "abs",
    "abs",
    "acos",
    "acosh",
    "add",
    "and_",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
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