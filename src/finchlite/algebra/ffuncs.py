# AI modified: 2026-04-08T22:22:21Z 84b3c0ad
import builtins
import operator
from functools import reduce
from typing import Any

import numpy as np

from .algebra import (
    COperator,
    FinchOperator,
    NumbaOperator,
    type_max,
    type_min,
)
from .ftypes import (
    FDType,
    FDTypeBoolean,
    FDTypeNumpyInteger,
    FDTypeOrdered,
    FType,
    TupleFType,
    bool,
    ftype,
    promote_type,
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


class CUnaryOperator(COperator):
    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        return f"{self.c_symbol}{ctx(args[0])}"


class NAryFinchOperator(FinchOperator):
    def return_type(self, *args) -> FType:  # type: ignore[override]
        new_args: list[Any] = []
        for arg in args:
            arg_type = ftype(arg)
            assert isinstance(arg_type, FDType)
            new_args.append(arg_type(True))
        return ftype(self(*new_args))


class BinaryFinchOperator(FinchOperator):
    def return_type(self, a: FType, b: FType) -> FType:  # type: ignore[override]
        assert isinstance(a, FDType) and isinstance(b, FDType)
        return ftype(self(a(True), b(True)))


class UnaryFinchOperator(FinchOperator):
    def return_type(self, a: FType) -> FType:  # type: ignore[override]
        assert isinstance(a, FDType)
        return ftype(self(a(True)))


class ComparisonFinchOperator(FinchOperator):
    def return_type(self, a: FType, b: FType) -> FType:  # type: ignore[override]
        assert isinstance(a, FDType) and isinstance(b, FDType)
        return bool


class _Add(NAryFinchOperator, CNAryOperator, NumbaOperator):
    is_associative = True
    is_commutative = True

    def __repr__(self) -> str:
        return "add"

    @property
    def c_symbol(self) -> str:
        return "+"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.add, args)

    def is_identity(self, arg: Any) -> builtins.bool:
        return arg == 0

    def is_annihilator(self, arg: Any, *argtypes: Any) -> builtins.bool:
        return np.isinf(arg)

    def repeat_operator(self):
        return mul

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return self(type_(0), type_(0))

    def numba_name(self) -> str:
        return "+"


add = _Add()


class _Mul(NAryFinchOperator, CNAryOperator, NumbaOperator):
    is_associative = True
    is_commutative = True

    def __repr__(self) -> str:
        return "mul"

    @property
    def c_symbol(self) -> str:
        return "*"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.mul, args)

    def is_identity(self, arg: Any) -> builtins.bool:
        return arg == 1

    def repeat_operator(self):
        return pow

    def is_distributive(self, other_op: "FinchOperator") -> builtins.bool:
        return isinstance(other_op, (_Add, _Sub))

    def is_annihilator(self, val, *argtypes: Any) -> builtins.bool:
        return val == 0

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return self(type_(1), type_(1))

    def numba_name(self) -> str:
        return "*"


mul = _Mul()


class _Sub(BinaryFinchOperator, CBinaryOperator, NumbaOperator):
    def __repr__(self) -> str:
        return "sub"

    @property
    def c_symbol(self) -> str:
        return "-"

    def __call__(self, a: Any, b: Any):
        return operator.sub(a, b)

    def numba_name(self) -> str:
        return "-"


sub = _Sub()


class _TrueDiv(BinaryFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "/"

    def __repr__(self) -> str:
        return "truediv"

    def __call__(self, a: Any, b: Any):
        return operator.truediv(a, b)

    def is_identity(self, arg):
        return arg == 1


truediv = _TrueDiv()


class _FloorDiv(BinaryFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "/"

    def __repr__(self) -> str:
        return "floordiv"

    def __call__(self, a: Any, b: Any):
        return operator.floordiv(a, b)


floordiv = _FloorDiv()


class _Mod(BinaryFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "%"

    def __repr__(self) -> str:
        return "mod"

    def __call__(self, a: Any, b: Any):
        return operator.mod(a, b)


mod = _Mod()


class _DivMod(BinaryFinchOperator):
    def __call__(self, a: Any, b: Any):
        return divmod(a, b)

    def __repr__(self) -> str:
        return "divmod"


divmod = _DivMod()


class _Pow(BinaryFinchOperator, COperator):
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

    def is_annihilator(self, arg, *argtypes: Any) -> builtins.bool:
        return arg == 0

    def __repr__(self) -> str:
        return "pow"


pow = _Pow()


class _LShift(BinaryFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "<<"

    def __call__(self, a: Any, b: Any):
        return operator.lshift(a, b)

    def is_identity(self, arg):
        return arg == 0

    def __repr__(self) -> str:
        return "lshift"


lshift = _LShift()


class _RShift(BinaryFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return ">>"

    def __call__(self, a: Any, b: Any):
        return operator.rshift(a, b)

    def is_identity(self, arg):
        return arg == 0

    def __repr__(self) -> str:
        return "rshift"


rshift = _RShift()


class _And(NAryFinchOperator, CNAryOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __repr__(self) -> str:
        return "and_"

    @property
    def c_symbol(self) -> str:
        return "&"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.and_, args)

    def is_identity(self, arg):
        return bool(arg)

    def is_annihilator(self, arg, *argtypes: Any) -> builtins.bool:
        return not bool(arg)

    def is_distributive(self, other_op: "FinchOperator") -> builtins.bool:
        return isinstance(other_op, (_Or, _Xor))

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return self(type_(True), type_(True))


and_ = _And()


class _Xor(NAryFinchOperator, CNAryOperator):
    is_associative = True
    is_commutative = True

    def __repr__(self) -> str:
        return "xor"

    @property
    def c_symbol(self) -> str:
        return "^"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.xor, args)

    def is_identity(self, arg):
        return arg == 0

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return self(type_(False), type_(False))


xor = _Xor()


class _Or(NAryFinchOperator, CNAryOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __repr__(self) -> str:
        return "or_"

    @property
    def c_symbol(self) -> str:
        return "|"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.or_, args)

    def is_identity(self, arg):
        return not bool(arg)

    def is_annihilator(self, arg, *argtypes: Any) -> builtins.bool:
        if not argtypes:
            return False
        try:
            promoted = reduce(promote_type, (ftype(arg), *argtypes))
        except TypeError:
            return False
        cast_arg = promoted(arg)
        if isinstance(promoted, FDTypeBoolean):
            return builtins.bool(cast_arg)
        if isinstance(promoted, FDTypeNumpyInteger):
            return cast_arg == ~promoted.dtype(0)
        return False

    def is_distributive(self, other_op: "FinchOperator") -> builtins.bool:
        return isinstance(other_op, _And)

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return self(type_(False), type_(False))


or_ = _Or()


class _Not(CUnaryOperator):
    @property
    def c_symbol(self) -> str:
        return "!"

    def __repr__(self) -> str:
        return "not_"


not_ = _Not()


class _Abs(UnaryFinchOperator):
    is_idempotent = True

    def __repr__(self) -> str:
        return "abs"

    def __call__(self, a: Any):
        return operator.abs(a)


abs = _Abs()


class _Pos(UnaryFinchOperator):
    is_idempotent = True

    def __repr__(self) -> str:
        return "pos"

    def __call__(self, a: Any):
        return operator.pos(a)


pos = _Pos()


class _Neg(UnaryFinchOperator):
    def __call__(self, a: Any):
        return operator.neg(a)

    def __repr__(self) -> str:
        return "neg"


neg = _Neg()


class _Invert(UnaryFinchOperator, CUnaryOperator):
    @property
    def c_symbol(self) -> str:
        return "~"

    def __call__(self, a: Any):
        return operator.invert(a)

    def __repr__(self) -> str:
        return "invert"


invert = _Invert()


class _Eq(ComparisonFinchOperator, CBinaryOperator, NumbaOperator):
    @property
    def c_symbol(self) -> str:
        return "=="

    def numba_name(self) -> str:
        return "=="

    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.eq(a, b)

    def __repr__(self) -> str:
        return "eq"


eq = _Eq()


class _Ne(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "!="

    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.ne(a, b)

    def __repr__(self) -> str:
        return "ne"


ne = _Ne()


class _Gt(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return ">"

    def __call__(self, a: Any, b: Any):
        return operator.gt(a, b)

    def __repr__(self) -> str:
        return "gt"


gt = _Gt()


class _Lt(ComparisonFinchOperator, CBinaryOperator, NumbaOperator):
    @property
    def c_symbol(self) -> str:
        return "<"

    def numba_name(self) -> str:
        return "<"

    def __call__(self, a: Any, b: Any):
        return operator.lt(a, b)

    def __repr__(self) -> str:
        return "lt"


lt = _Lt()


class _Ge(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return ">="

    def __call__(self, a: Any, b: Any):
        return operator.ge(a, b)

    def __repr__(self) -> str:
        return "ge"


ge = _Ge()


class _Le(ComparisonFinchOperator, CBinaryOperator):
    @property
    def c_symbol(self) -> str:
        return "<="

    def __call__(self, a: Any, b: Any):
        return operator.le(a, b)

    def __repr__(self) -> str:
        return "le"


le = _Le()


class _Divide(BinaryFinchOperator):
    def __call__(self, a, b):
        return np.divide(a, b)

    def is_identity(self, val) -> builtins.bool:
        return val == 1

    def __repr__(self) -> str:
        return "divide"


divide = _Divide()


class _LogAddExp(BinaryFinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = False

    def __call__(self, a, b):
        return np.logaddexp(a, b)

    def is_identity(self, val) -> builtins.bool:
        return val == -np.inf

    def is_annihilator(self, val, *argtypes: Any) -> builtins.bool:
        return val == np.inf

    def init_value(self, type_: FType) -> Any:
        return -np.inf

    def __repr__(self) -> str:
        return "logaddexp"


logaddexp = _LogAddExp()


class _LogicalAnd(BinaryFinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return np.logical_and(a, b)

    def is_identity(self, val) -> builtins.bool:
        return builtins.bool(val)

    def is_annihilator(self, val, *argtypes: Any) -> builtins.bool:
        return not builtins.bool(val)

    def is_distributive(self, other_op: FinchOperator) -> builtins.bool:
        return isinstance(other_op, (_LogicalOr, _LogicalXor))

    def init_value(self, type_: FType) -> Any:
        return True

    def __repr__(self) -> str:
        return "logical_and"


logical_and = _LogicalAnd()


class _LogicalOr(BinaryFinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, a, b):
        return np.logical_or(a, b)

    def is_identity(self, val) -> builtins.bool:
        return not builtins.bool(val)

    def is_annihilator(self, val, *argtypes: Any) -> builtins.bool:
        return builtins.bool(val)

    def is_distributive(self, other_op: FinchOperator) -> builtins.bool:
        return isinstance(other_op, _LogicalAnd)

    def init_value(self, type_: FType) -> Any:
        return False

    def __repr__(self) -> str:
        return "logical_or"


logical_or = _LogicalOr()


class _LogicalXor(BinaryFinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = False

    def __call__(self, a, b):
        return np.logical_xor(a, b)

    def is_identity(self, val) -> builtins.bool:
        return not builtins.bool(val)

    def init_value(self, type_: FType) -> Any:
        return False

    def __repr__(self) -> str:
        return "logical_xor"


logical_xor = _LogicalXor()


class _LogicalNot(UnaryFinchOperator):
    def __call__(self, a):
        return np.logical_not(a)

    def __repr__(self) -> str:
        return "logical_not"


logical_not = _LogicalNot()


class _Truth(UnaryFinchOperator):
    def __call__(self, a: Any):
        return bool(a)

    def __repr__(self) -> str:
        return "truth"


truth = _Truth()


class _Min(NAryFinchOperator, NumbaOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, *args: Any) -> Any:
        def op(a, b):
            A = ftype(a)
            B = ftype(b)
            assert isinstance(A, FDType) and isinstance(B, FDType)
            C = promote_type(A, B)
            return C(builtins.min(a, b))

        return reduce(op, args)

    def is_identity(self, val) -> builtins.bool:
        return val == np.inf

    def init_value(self, type_: FType):
        assert isinstance(type_, FDTypeOrdered)
        return type_max(type_)

    def numba_name(self) -> str:
        return "min"

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return f"min({', '.join(map(ctx, args))})"

    def __repr__(self) -> str:
        return "min"


min = _Min()


class _Max(NAryFinchOperator, NumbaOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, *args: Any) -> Any:
        def op(a, b):
            A = ftype(a)
            B = ftype(b)
            assert isinstance(A, FDType) and isinstance(B, FDType)
            C = promote_type(A, B)
            return C(builtins.max(a, b))

        return reduce(op, args)

    def is_identity(self, val) -> builtins.bool:
        return val == -np.inf

    def init_value(self, type_: FType):
        assert isinstance(type_, FDTypeOrdered)
        return type_min(type_)

    def numba_name(self) -> str:
        return "max"

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return f"max({', '.join(map(ctx, args))})"

    def __repr__(self) -> str:
        return "max"


max = _Max()


class _Remainder(BinaryFinchOperator):
    def __call__(self, a, b):
        return np.remainder(a, b)

    def __repr__(self) -> str:
        return "remainder"


remainder = _Remainder()


class _Hypot(BinaryFinchOperator):
    is_commutative = True

    def __call__(self, a, b):
        return np.hypot(a, b)

    def __repr__(self) -> str:
        return "hypot"


hypot = _Hypot()


class _Atan2(BinaryFinchOperator):
    def __call__(self, a, b):
        return np.atan2(a, b)

    def __repr__(self) -> str:
        return "atan2"


atan2 = _Atan2()


class _Copysign(BinaryFinchOperator):
    def __call__(self, a, b):
        return np.copysign(a, b)

    def __repr__(self) -> str:
        return "copysign"


copysign = _Copysign()


class _Nextafter(BinaryFinchOperator):
    def __call__(self, a, b):
        return np.nextafter(a, b)

    def __repr__(self) -> str:
        return "nextafter"


nextafter = _Nextafter()


class _IsFinite(UnaryFinchOperator):
    def __call__(self, a):
        return np.isfinite(a)

    def __repr__(self) -> str:
        return "isfinite"


isfinite = _IsFinite()


class _IsInf(UnaryFinchOperator):
    def __call__(self, a):
        return np.isinf(a)

    def __repr__(self) -> str:
        return "isinf"


isinf = _IsInf()


class _IsNan(UnaryFinchOperator):
    def __call__(self, a):
        return np.isnan(a)

    def __repr__(self) -> str:
        return "isnan"


isnan = _IsNan()


class _IsComplexObj(UnaryFinchOperator):
    def __call__(self, a):
        return np.iscomplexobj(a)

    def __repr__(self) -> str:
        return "iscomplexobj"


iscomplexobj = _IsComplexObj()


class _Real(UnaryFinchOperator):
    def __call__(self, a):
        return np.real(a)

    def return_type(self, a: FType) -> FType:  # type: ignore[override]
        return ftype(float)

    def __repr__(self) -> str:
        return "real"


real = _Real()


class _Imag(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.imag(a)

    def return_type(self, a: FType) -> FType:  # type: ignore[override]
        return ftype(float)

    def __repr__(self) -> str:
        return "imag"


imag = _Imag()


class _Clip(FinchOperator):
    def __call__(self, a: Any, b: Any, c: Any):
        return np.clip(a, b, c)

    def return_type(self, a: FType, b: FType, c: FType) -> FType:  # type: ignore[override]
        return ftype(float)

    def __repr__(self) -> str:
        return "clip"


clip = _Clip()


class _Equal(BinaryFinchOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.equal(a, b)

    def __repr__(self) -> str:
        return "equal"


equal = _Equal()


class _NotEqual(BinaryFinchOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.not_equal(a, b)

    def __repr__(self) -> str:
        return "not_equal"


not_equal = _NotEqual()


class _Less(BinaryFinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.less(a, b)

    def __repr__(self) -> str:
        return "less"


less = _Less()


class _LessEqual(BinaryFinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.less_equal(a, b)

    def __repr__(self) -> str:
        return "less_equal"


less_equal = _LessEqual()


class _Greater(BinaryFinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater(a, b)

    def __repr__(self) -> str:
        return "greater"


greater = _Greater()


class _GreaterEqual(BinaryFinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater_equal(a, b)

    def __repr__(self) -> str:
        return "greater_equal"


greater_equal = _GreaterEqual()


class _Reciprocal(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.reciprocal(a)

    def __repr__(self) -> str:
        return "reciprocal"


reciprocal = _Reciprocal()


class _Sin(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.sin(a)

    def __repr__(self) -> str:
        return "sin"


sin = _Sin()


class _Cos(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.cos(a)

    def __repr__(self) -> str:
        return "cos"


cos = _Cos()


class _Tan(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.tan(a)

    def __repr__(self) -> str:
        return "tan"


tan = _Tan()


class _Sinh(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.sinh(a)

    def __repr__(self) -> str:
        return "sinh"


sinh = _Sinh()


class _Cosh(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.cosh(a)

    def __repr__(self) -> str:
        return "cosh"


cosh = _Cosh()


class _Tanh(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.tanh(a)

    def __repr__(self) -> str:
        return "tanh"


tanh = _Tanh()


class _Atan(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.atan(a)

    def __repr__(self) -> str:
        return "atan"


atan = _Atan()


class _Asinh(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.asinh(a)

    def __repr__(self) -> str:
        return "asinh"


asinh = _Asinh()


class _Asin(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.asin(a)

    def __repr__(self) -> str:
        return "asin"


asin = _Asin()


class _Acos(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.acos(a)

    def __repr__(self) -> str:
        return "acos"


acos = _Acos()


class _Acosh(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.acosh(a)

    def __repr__(self) -> str:
        return "acosh"


acosh = _Acosh()


class _Atanh(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.atanh(a)

    def __repr__(self) -> str:
        return "atanh"


atanh = _Atanh()

# Backward-compatible aliases.
arcsin = asin
arccos = acos
arctan = atan
arcsinh = asinh
arccosh = acosh
arctanh = atanh


class _Round(UnaryFinchOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.round(a)

    def __repr__(self) -> str:
        return "round"


round = _Round()


class _Floor(UnaryFinchOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.floor(a)

    def __repr__(self) -> str:
        return "floor"


floor = _Floor()


class _Ceil(UnaryFinchOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.ceil(a)

    def __repr__(self) -> str:
        return "ceil"


ceil = _Ceil()


class _Trunc(UnaryFinchOperator):
    is_idempotent = True

    def __call__(self, a: Any):
        return np.trunc(a)

    def __repr__(self) -> str:
        return "trunc"


trunc = _Trunc()


class _Exp(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.exp(a)

    def __repr__(self) -> str:
        return "exp"


exp = _Exp()


class _Expm1(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.expm1(a)

    def __repr__(self) -> str:
        return "expm1"


expm1 = _Expm1()


class _Log(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.log(a)

    def __repr__(self) -> str:
        return "log"


log = _Log()


class _Log1p(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.log1p(a)

    def __repr__(self) -> str:
        return "log1p"


log1p = _Log1p()


class _Log2(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.log2(a)

    def __repr__(self) -> str:
        return "log2"


log2 = _Log2()


class _Log10(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.log10(a)

    def __repr__(self) -> str:
        return "log10"


log10 = _Log10()


class _Signbit(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.signbit(a)

    def __repr__(self) -> str:
        return "signbit"


signbit = _Signbit()


class _Sqrt(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.sqrt(a)

    def __repr__(self) -> str:
        return "sqrt"


sqrt = _Sqrt()


class _Square(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.square(a)

    def __repr__(self) -> str:
        return "square"


square = _Square()


class _Sign(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.sign(a)

    def __repr__(self) -> str:
        return "sign"


sign = _Sign()


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

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        return y

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        return ctx(args[1])

    def __repr__(self) -> str:
        return "_initwrite"


def init_write(value):
    return _InitWrite(value)


class _Overwrite(FinchOperator):
    """
    Overwrite(x, y) returns y always.
    """

    def __call__(self, x: Any, y: Any):
        return y

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        return y

    def __repr__(self) -> str:
        return "overwrite"


overwrite = _Overwrite()


class _FirstArg(FinchOperator):
    """
    Returns the first argument passed to it.
    """

    def __call__(self, *args):
        return args[0] if args else None

    def return_type(self, *args: FType) -> FType:
        return args[0]

    def __repr__(self) -> str:
        return "first_arg"


first_arg = _FirstArg()


class _Identity(FinchOperator):
    """
    Returns the input value unchanged.
    """

    is_idempotent = True

    def __call__(self, x: Any):
        return x

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        return x

    def __repr__(self) -> str:
        return "identity"


identity = _Identity()


class _Conjugate(FinchOperator):
    """
    Returns the complex conjugate of the input value.
    """

    def __call__(self, x: Any):
        return np.conjugate(x)

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        return x

    def __repr__(self) -> str:
        return "conjugate"


conjugate = _Conjugate()


class _MakeTuple(FinchOperator, NumbaOperator):
    is_commutative = False
    is_associative = False

    def __call__(self, *args: Any) -> tuple:
        return tuple(args)

    def return_type(self, *args: FType) -> FType:
        return TupleFType.from_tuple(args)

    def numba_literal(self, val: Any, ctx: Any, *args: Any):
        return f"({','.join([ctx(arg) for arg in args])},)"

    def __repr__(self) -> str:
        return "make_tuple"


make_tuple = _MakeTuple()


class _Scansearch(FinchOperator, NumbaOperator):
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
        hi = builtins.min(p, hi) + u  # type: ignore[call-overload]

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

    def return_type(self, arr: FType, x: FType, lo: FType, hi: FType) -> FType:  # type: ignore[override]
        return hi

    def numba_literal(self, val: Any, ctx: Any, *args: Any) -> Any:
        arr = args[0]
        x = args[1]
        lo = args[2]
        hi = args[3]
        return f"scansearch({ctx(arr)}, {ctx(x)}, {ctx(lo)}, {ctx(hi)})"

    def __repr__(self) -> str:
        return "scansearch"


scansearch = _Scansearch()

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
    "max",
    "max",
    "min",
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
