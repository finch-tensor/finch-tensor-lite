import builtins
import operator
from functools import reduce
from typing import Any

import numpy as np

from .algebra import (
    FinchOperator,
    type_max,
    type_min,
)
from .ftypes import (
    FDType,
    FDTypeBoolean,
    FDTypeInteger,
    FDTypeOrdered,
    FDTypeUnsignedInteger,
    FType,
    TupleFType,
    bool,
    ftype,
    int64,
    promote_type,
    uint64,
)


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


class _Add(NAryFinchOperator):
    is_associative = True
    is_commutative = True

    def __repr__(self) -> str:
        return "add"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.add, args)

    def is_identity(self, arg: Any) -> builtins.bool:
        return arg == 0

    def is_annihilator(self, arg: Any) -> builtins.bool:
        try:
            return np.isinf(arg)
        except (TypeError, ValueError):
            # If arg is not a type that can be checked for infinity, it cannot
            # be an annihilator for addition.
            return False

    def repeat_operator(self):
        return mul

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        if isinstance(type_, FDTypeInteger) and not isinstance(type_, FDTypeBoolean):
            if isinstance(type_, FDTypeUnsignedInteger):
                return self(type_(0), uint64(0))
            return self(type_(0), int64(0))
        return type_(0)


add = _Add()


class _Mul(NAryFinchOperator):
    is_associative = True
    is_commutative = True

    def __repr__(self) -> str:
        return "mul"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.mul, args)

    def is_identity(self, arg: Any) -> builtins.bool:
        return arg == 1

    def repeat_operator(self):
        return pow

    def is_distributive(self, other_op: "FinchOperator") -> builtins.bool:
        return isinstance(other_op, _Add | _Sub)

    def is_annihilator(self, val):
        return val == 0

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        if isinstance(type_, FDTypeInteger) and not isinstance(type_, FDTypeBoolean):
            if isinstance(type_, FDTypeUnsignedInteger):
                return self(type_(1), uint64(1))
            return self(type_(1), int64(1))
        return type_(1)


mul = _Mul()


class _Sub(BinaryFinchOperator):
    def __repr__(self) -> str:
        return "sub"

    def __call__(self, a: Any, b: Any):
        return operator.sub(a, b)


sub = _Sub()


class _TrueDiv(BinaryFinchOperator):
    def __repr__(self) -> str:
        return "truediv"

    def __call__(self, a: Any, b: Any):
        return np.true_divide(a, b)

    def is_identity(self, arg):
        return arg == 1


truediv = _TrueDiv()


class _FloorDiv(BinaryFinchOperator):
    def __repr__(self) -> str:
        return "floor_divide"

    def __call__(self, a: Any, b: Any):
        return np.floor_divide(a, b)


floordiv = _FloorDiv()


class _Mod(BinaryFinchOperator):
    def __repr__(self) -> str:
        return "mod"

    def __call__(self, a: Any, b: Any):
        return np.mod(a, b)


mod = _Mod()


class _DivMod(BinaryFinchOperator):
    def __call__(self, a: Any, b: Any):
        return divmod(a, b)

    def __repr__(self) -> str:
        return "divmod"


divmod = _DivMod()


class _Pow(BinaryFinchOperator):
    @property
    def c_symbol(self) -> str:
        return "pow"

    def __call__(self, a: Any, b: Any):
        return np.power(a, b)

    def is_identity(self, arg):
        return arg == 1

    def is_annihilator(self, arg):
        return arg == 0

    def __repr__(self) -> str:
        return "pow"


pow = _Pow()


class _LShift(BinaryFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.lshift(a, b)

    def is_identity(self, arg):
        return arg == 0

    def __repr__(self) -> str:
        return "lshift"


lshift = _LShift()


class _RShift(BinaryFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.rshift(a, b)

    def is_identity(self, arg):
        return arg == 0

    def __repr__(self) -> str:
        return "rshift"


rshift = _RShift()


class _And(NAryFinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __repr__(self) -> str:
        return "and_"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.and_, args)

    def is_identity(self, arg):
        return bool(arg)

    def is_annihilator(self, arg):
        return not bool(arg)

    def is_distributive(self, other_op: "FinchOperator") -> builtins.bool:
        return isinstance(other_op, _Or | _Xor)

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return self(type_(True), type_(True))


and_ = _And()


class _Xor(NAryFinchOperator):
    is_associative = True
    is_commutative = True

    def __repr__(self) -> str:
        return "xor"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.xor, args)

    def is_identity(self, arg):
        return arg == 0

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return self(type_(False), type_(False))


xor = _Xor()


class _Or(NAryFinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __repr__(self) -> str:
        return "or_"

    def __call__(self, *args: Any) -> Any:
        return reduce(operator.or_, args)

    def is_identity(self, arg):
        return not bool(arg)

    def is_annihilator(self, arg):
        return bool(arg)

    def is_distributive(self, other_op: "FinchOperator") -> builtins.bool:
        return isinstance(other_op, _And)

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return self(type_(False), type_(False))


or_ = _Or()


class _Not(UnaryFinchOperator):
    def __call__(self, a: Any):
        return operator.not_(a)

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


class _Invert(UnaryFinchOperator):
    def __call__(self, a: Any):
        return operator.invert(a)

    def __repr__(self) -> str:
        return "invert"


invert = _Invert()


class _Eq(ComparisonFinchOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.eq(a, b)

    def __repr__(self) -> str:
        return "eq"


eq = _Eq()


class _Ne(ComparisonFinchOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return operator.ne(a, b)

    def __repr__(self) -> str:
        return "ne"


ne = _Ne()


class _Gt(ComparisonFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.gt(a, b)

    def __repr__(self) -> str:
        return "gt"


gt = _Gt()


class _Lt(ComparisonFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.lt(a, b)

    def __repr__(self) -> str:
        return "lt"


lt = _Lt()


class _Ge(ComparisonFinchOperator):
    def __call__(self, a: Any, b: Any):
        return operator.ge(a, b)

    def __repr__(self) -> str:
        return "ge"


ge = _Ge()


class _Le(ComparisonFinchOperator):
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

    def is_annihilator(self, val) -> builtins.bool:
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

    def is_annihilator(self, val) -> builtins.bool:
        return not builtins.bool(val)

    def is_distributive(self, other_op: FinchOperator) -> builtins.bool:
        return isinstance(other_op, _LogicalOr | _LogicalXor)

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

    def is_annihilator(self, val) -> builtins.bool:
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


class _Min(NAryFinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, *args: Any) -> Any:
        def op(a, b):
            A = ftype(a)
            B = ftype(b)
            assert isinstance(A, FDType) and isinstance(B, FDType)
            C = promote_type(A, B)
            return C(np.minimum(a, b))

        return reduce(op, args)

    def is_identity(self, val) -> builtins.bool:
        return val == np.inf

    def init_value(self, type_: FType):
        assert isinstance(type_, FDTypeOrdered)
        return type_max(type_)

    def __repr__(self) -> str:
        return "min"


min = _Min()


class _Max(NAryFinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, *args: Any) -> Any:
        def op(a, b):
            A = ftype(a)
            B = ftype(b)
            assert isinstance(A, FDType) and isinstance(B, FDType)
            C = promote_type(A, B)
            return C(np.maximum(a, b))

        return reduce(op, args)

    def is_identity(self, val) -> builtins.bool:
        return val == -np.inf

    def init_value(self, type_: FType):
        assert isinstance(type_, FDTypeOrdered)
        return type_min(type_)

    def __repr__(self) -> str:
        return "max"


max = _Max()


class _MinBy(FinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, x: tuple, y: tuple) -> tuple:
        x_key, y_key = x[0], y[0]
        x_last, y_last = x[-1], y[-1]
        if x_key < y_key:
            return x
        if y_key < x_key:
            return y
        return x if x_last <= y_last else y

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType) and isinstance(y, TupleFType)
        if len(x.struct_fieldtypes) != len(y.struct_fieldtypes):
            raise TypeError("Tuple operands must have the same length.")
        return TupleFType.from_tuple(
            tuple(
                promote_type(x_type, y_type)
                for x_type, y_type in zip(
                    x.struct_fieldtypes, y.struct_fieldtypes, strict=True
                )
            )
        )

    def __repr__(self) -> str:
        return "minby"


minby = _MinBy()


class _MaxBy(FinchOperator):
    is_associative = True
    is_commutative = True
    is_idempotent = True

    def __call__(self, x: tuple, y: tuple) -> tuple:
        x_key, y_key = x[0], y[0]
        x_last, y_last = x[-1], y[-1]
        if x_key > y_key:
            return x
        if y_key > x_key:
            return y
        return x if x_last <= y_last else y

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType) and isinstance(y, TupleFType)
        if len(x.struct_fieldtypes) != len(y.struct_fieldtypes):
            raise TypeError("Tuple operands must have the same length.")
        return TupleFType.from_tuple(
            tuple(
                promote_type(x_type, y_type)
                for x_type, y_type in zip(
                    x.struct_fieldtypes, y.struct_fieldtypes, strict=True
                )
            )
        )

    def __repr__(self) -> str:
        return "maxby"


maxby = _MaxBy()


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


class _Conj(UnaryFinchOperator):
    def __call__(self, a: Any):
        return np.conj(a)

    def __repr__(self) -> str:
        return "conj"


conj = _Conj()


class _Clip(FinchOperator):
    def __call__(self, a: Any, b: Any, c: Any):
        return ftype(a)(np.clip(a, b, c))

    def return_type(self, a: FType, b: FType, c: FType) -> FType:  # type: ignore[override]
        return a

    def __repr__(self) -> str:
        return "clip"


clip = _Clip()


class _Cast(FinchOperator):
    def __init__(self, dtype: FType):
        self.dtype = dtype

    def __call__(self, a: Any):
        assert isinstance(self.dtype, FDType)
        return self.dtype(a)

    def return_type(self, a: FType) -> FType:  # type: ignore[override]
        return self.dtype

    def __eq__(self, other):
        return isinstance(other, _Cast) and self.dtype == other.dtype

    def __hash__(self):
        return hash((type(self), self.dtype))

    def __repr__(self) -> str:
        return "astype"


def astype(dtype: FType):
    return _Cast(dtype)


class _Equal(ComparisonFinchOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.equal(a, b)

    def __repr__(self) -> str:
        return "equal"


equal = _Equal()


class _Same(BinaryFinchOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        same_method = getattr(a, "__same__", None)
        if same_method is not None:
            res = same_method(b)
            if res is not NotImplemented:
                return res
        rsame_method = getattr(b, "__rsame__", None)
        if rsame_method is not None:
            res = rsame_method(a)
            if res is not NotImplemented:
                return res
        try:
            return np.logical_or(
                np.equal(a, b), np.logical_and(np.isnan(a), np.isnan(b))
            )
        except TypeError:
            return np.equal(a, b)

    def __repr__(self) -> str:
        return "same"


same = _Same()


def samehash(a: Any):
    samehash_method = getattr(a, "__samehash__", None)
    if samehash_method is not None:
        res = samehash_method()
        if res is not NotImplemented:
            return res
    if np.all(same(a, a)) and not np.array_equal(a, a):
        return ("nan", ftype(a))
    return a


class _NotSame(BinaryFinchOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.logical_not(same(a, b))

    def __repr__(self) -> str:
        return "not_same"


not_same = _NotSame()


class _NotEqual(ComparisonFinchOperator):
    is_commutative = True

    def __call__(self, a: Any, b: Any):
        return np.not_equal(a, b)

    def __repr__(self) -> str:
        return "not_equal"


not_equal = _NotEqual()


class _Less(ComparisonFinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.less(a, b)

    def __repr__(self) -> str:
        return "less"


less = _Less()


class _LessEqual(ComparisonFinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.less_equal(a, b)

    def __repr__(self) -> str:
        return "less_equal"


less_equal = _LessEqual()


class _Greater(ComparisonFinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater(a, b)

    def __repr__(self) -> str:
        return "greater"


greater = _Greater()


class _GreaterEqual(ComparisonFinchOperator):
    def __call__(self, a: Any, b: Any):
        return np.greater_equal(a, b)

    def __repr__(self) -> str:
        return "greater_equal"


class _Where(FinchOperator):
    def __call__(self, a: Any, b: Any, c: Any):
        res = np.where(a, b, c)
        if isinstance(res, np.ndarray) and res.shape == ():
            return res[()]
        return res

    def return_type(self, cond: FDType, x1: FDType, x2: FDType) -> FDType:  # type: ignore[override]
        return promote_type(x1, x2)

    def __repr__(self) -> str:
        return "where"


where = _Where()

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


class _InitWrite(FinchOperator):
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


class _Choose(FinchOperator):
    is_associative = True

    def __init__(self, fill_value):
        self.fill_value = fill_value

    def __eq__(self, other):
        return isinstance(other, _Choose) and builtins.bool(
            np.all(same(self.fill_value, other.fill_value))
        )

    def __hash__(self):
        return hash((type(self), samehash(self.fill_value)))

    def __call__(self, *args: Any) -> Any:
        for arg in args:
            if not np.all(same(arg, self.fill_value)):
                return arg
        return self.fill_value

    def return_type(self, *args: FType) -> FType:
        if not args:
            return ftype(self.fill_value)
        result_arg = args[0]
        assert isinstance(result_arg, FDType)
        result: FDType = result_arg
        for arg in args[1:]:
            assert isinstance(arg, FDType)
            result = promote_type(result, arg)
        return result

    def is_identity(self, val: Any) -> builtins.bool:
        return builtins.bool(np.all(same(val, self.fill_value)))

    def init_value(self, type_: FType) -> Any:
        assert isinstance(type_, FDType)
        return type_(self.fill_value)

    def __repr__(self) -> str:
        return f"choose({self.fill_value!r})"


def choose(fill_value):
    return _Choose(fill_value)


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


class _MakeTuple(FinchOperator):
    is_commutative = False
    is_associative = False

    def __call__(self, *args: Any) -> tuple:
        return tuple(args)

    def return_type(self, *args: FType) -> FType:
        return TupleFType.from_tuple(args)

    def __repr__(self) -> str:
        return "make_tuple"


make_tuple = _MakeTuple()


class _Last(FinchOperator):
    def __call__(self, x: tuple) -> Any:
        return x[-1]

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType)
        return x.struct_fieldtypes[-1]

    def __repr__(self) -> str:
        return "last"


last = _Last()


class _ScaledSquare(FinchOperator):
    def __call__(self, x: Any) -> tuple:
        if x == 0:
            return (np.true_divide(type(x)(0), type(x)(1)), x)
        return (np.true_divide(type(x)(1), type(x)(1)), x)

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, FDType)
        return TupleFType.from_tuple((truediv.return_type(x, x), x))

    def __repr__(self) -> str:
        return "scaled_square"


scaled_square = _ScaledSquare()


class _ScaledPower(FinchOperator):
    def __init__(self, exponent: float):
        self.exponent = exponent

    def __call__(self, x: Any) -> tuple:
        return scaled_square(x)

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        return scaled_square.return_type(x)

    def __eq__(self, other):
        return isinstance(other, _ScaledPower) and self.exponent == other.exponent

    def __hash__(self):
        return hash((type(self), self.exponent))

    def __repr__(self) -> str:
        return f"scaled_power({self.exponent!r})"


def scaled_power(exponent: float):
    if exponent == 2.0:
        return scaled_square
    return _ScaledPower(exponent)


class _AddScaledPower(FinchOperator):
    is_associative = True
    is_commutative = True

    def __init__(self, exponent: float):
        self.exponent = exponent

    def __call__(self, x: tuple, y: tuple) -> tuple:
        x_arg, x_scale = x
        y_arg, y_scale = y
        if x_scale < y_scale:
            x_arg, y_arg = y_arg, x_arg
            x_scale, y_scale = y_scale, x_scale
        if x_scale > y_scale:
            return (
                x_arg
                + y_arg * np.power(np.true_divide(y_scale, x_scale), self.exponent),
                x_scale,
            )
        return (x_arg + y_arg, x_scale)

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType) and isinstance(y, TupleFType)
        if len(x.struct_fieldtypes) != 2 or len(y.struct_fieldtypes) != 2:
            raise TypeError("Scaled power operands must be 2-tuples.")
        x_arg, x_scale = x.struct_fieldtypes
        y_arg, y_scale = y.struct_fieldtypes
        assert (
            isinstance(x_arg, FDType)
            and isinstance(x_scale, FDType)
            and isinstance(y_arg, FDType)
            and isinstance(y_scale, FDType)
        )
        return TupleFType.from_tuple(
            (promote_type(x_arg, y_arg), promote_type(x_scale, y_scale))
        )

    def is_identity(self, val: Any) -> builtins.bool:
        return builtins.bool(val[0] == 0 and val[1] == 0)

    def __eq__(self, other):
        return isinstance(other, _AddScaledPower) and self.exponent == other.exponent

    def __hash__(self):
        return hash((type(self), self.exponent))

    def __repr__(self) -> str:
        return f"add_scaled_power({self.exponent!r})"


class _AddScaledSquare(FinchOperator):
    is_associative = True
    is_commutative = True

    def __call__(self, x: tuple, y: tuple) -> tuple:
        x_arg, x_scale = x
        y_arg, y_scale = y
        if x_scale < y_scale:
            x_arg, y_arg = y_arg, x_arg
            x_scale, y_scale = y_scale, x_scale
        if x_scale > y_scale:
            ratio = np.true_divide(y_scale, x_scale)
            return (x_arg + y_arg * ratio * ratio, x_scale)
        return (x_arg + y_arg, x_scale)

    def return_type(self, x: FType, y: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType) and isinstance(y, TupleFType)
        if len(x.struct_fieldtypes) != 2 or len(y.struct_fieldtypes) != 2:
            raise TypeError("Scaled square operands must be 2-tuples.")
        x_arg, x_scale = x.struct_fieldtypes
        y_arg, y_scale = y.struct_fieldtypes
        assert (
            isinstance(x_arg, FDType)
            and isinstance(x_scale, FDType)
            and isinstance(y_arg, FDType)
            and isinstance(y_scale, FDType)
        )
        return TupleFType.from_tuple(
            (promote_type(x_arg, y_arg), promote_type(x_scale, y_scale))
        )

    def is_identity(self, val: Any) -> builtins.bool:
        return builtins.bool(val[0] == 0 and val[1] == 0)

    def __repr__(self) -> str:
        return "add_scaled_square"


add_scaled_square = _AddScaledSquare()


def add_scaled_power(exponent: float):
    if exponent == 2.0:
        return add_scaled_square
    return _AddScaledPower(exponent)


class _RootScaledPower(FinchOperator):
    def __init__(self, exponent: float):
        self.exponent = exponent

    def __call__(self, x: tuple) -> Any:
        arg, scale = x
        return np.power(arg, 1.0 / self.exponent) * scale

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType)
        if len(x.struct_fieldtypes) != 2:
            raise TypeError("Scaled power roots must be taken from 2-tuples.")
        arg, scale = x.struct_fieldtypes
        assert isinstance(arg, FDType) and isinstance(scale, FDType)
        return mul.return_type(pow.return_type(arg, ftype(float)), scale)

    def __eq__(self, other):
        return isinstance(other, _RootScaledPower) and self.exponent == other.exponent

    def __hash__(self):
        return hash((type(self), self.exponent))

    def __repr__(self) -> str:
        return f"root_scaled_power({self.exponent!r})"


class _RootScaledSquare(FinchOperator):
    def __call__(self, x: tuple) -> Any:
        arg, scale = x
        return np.sqrt(arg) * scale

    def return_type(self, x: FType) -> FType:  # type: ignore[override]
        assert isinstance(x, TupleFType)
        if len(x.struct_fieldtypes) != 2:
            raise TypeError("Scaled square roots must be taken from 2-tuples.")
        arg, scale = x.struct_fieldtypes
        assert isinstance(arg, FDType) and isinstance(scale, FDType)
        return mul.return_type(sqrt.return_type(arg), scale)

    def __repr__(self) -> str:
        return "root_scaled_square"


root_scaled_square = _RootScaledSquare()


def root_scaled_power(exponent: float):
    if exponent == 2.0:
        return root_scaled_square
    return _RootScaledPower(exponent)


class _Scansearch(FinchOperator):
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

    def __repr__(self) -> str:
        return "scansearch"


scansearch = _Scansearch()


class _ResizeIfSmaller(FinchOperator):
    """
    ResizeIfSmaller resizes an array to a new size if the new size is larger
    than the current size.

    It takes an array `arr` and a new size `new_size`, and returns a resized
    version of `arr` if `new_size` is larger than the current size of `arr`.
    If `new_size` is less than or equal to the current size of `arr`, it
    returns `arr` unchanged.
    """

    @staticmethod
    def _func(
        arr: np.ndarray, new_size: np.integer, fill_value: np.number
    ) -> np.ndarray:
        if new_size > arr.size:
            new_arr = np.full(new_size, fill_value, arr.dtype)
            new_arr[: arr.size] = arr
            return new_arr
        return arr

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def return_type(self, arr: FType, new_size: FType, fill_value: FType) -> FType:  # type: ignore[override]
        return arr

    def __repr__(self) -> str:
        return "resize_if_smaller"


resize_if_smaller = _ResizeIfSmaller()


__all__ = [
    "abs",
    "abs",
    "acos",
    "acosh",
    "add",
    "add_scaled_power",
    "add_scaled_square",
    "and_",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "asin",
    "asinh",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "choose",
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
    "last",
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
    "maxby",
    "min",
    "min",
    "minby",
    "mod",
    "mul",
    "ne",
    "neg",
    "nextafter",
    "not_equal",
    "not_same",
    "or_",
    "overwrite",
    "pos",
    "pow",
    "real",
    "reciprocal",
    "remainder",
    "resize_if_smaller",
    "round",
    "root_scaled_power",
    "root_scaled_square",
    "rshift",
    "same",
    "samehash",
    "scansearch",
    "sign",
    "signbit",
    "scaled_power",
    "scaled_square",
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
