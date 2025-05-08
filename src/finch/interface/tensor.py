from abc import ABC, abstractmethod
from . import lazy
from .fuse import *

class EagerTensor(ABC):
    @abstractmethod
    def shape(self):
        """Return the shape of the tensor."""
        pass

    @abstractmethod
    def dtype(self):
        """Return the data type of the tensor."""
        pass

    @abstractmethod
    def to_numpy(self):
        """Convert the tensor to a NumPy array."""
        pass

    @abstractmethod
    def __add__(self, other):
        lazy.compute(lazy.lazy(self).__add__(other))

    @abstractmethod
    def __sub__(self, other):
        """Define subtraction for tensors."""
        lazy.compute(lazy.lazy(self).__sub__(other))

    @abstractmethod
    def __mul__(self, other):
        """Define multiplication for tensors."""
        lazy.compute(lazy.lazy(self).__mul__(other))
    
    @abstractmethod
    def __matmul__(self, other):
        """Define matrix multiplication for tensors."""
        lazy.compute(lazy.lazy(self).__matmul__(other))
    
    @abstractmethod
    def __truediv__(self, other):
        """Define true division for tensors."""
        lazy.compute(lazy.lazy(self).__truediv__(other))
    
    @abstractmethod
    def __floordiv__(self, other):
        """Define floor division for tensors."""
        lazy.compute(lazy.lazy(self).__floordiv__(other))

    @abstractmethod
    def __mod__(self, other):
        """Define modulo for tensors."""
        lazy.compute(lazy.lazy(self).__mod__(other))

    @abstractmethod
    def __divmod__(self, other):
        """Define division modulo for tensors."""
        lazy.compute(lazy.lazy(self).__divmod__(other))
    
    @abstractmethod
    def __pow__(self, other):
        """Define power for tensors."""
        lazy.compute(lazy.lazy(self).__pow__(other))

    @abstractmethod
    def __lshift__(self, other):
        """Define left shift for tensors."""
        lazy.compute(lazy.lazy(self).__lshift__(other))

    @abstractmethod
    def __rshift__(self, other):
        """Define right shift for tensors."""
        lazy.compute(lazy.lazy(self).__rshift__(other))

    @abstractmethod
    def __and__(self, other):
        """Define bitwise AND for tensors."""
        lazy.compute(lazy.lazy(self).__and__(other))
    
    @abstractmethod
    def __or__(self, other):
        """Define bitwise OR for tensors."""
        lazy.compute(lazy.lazy(self).__or__(other))

    @abstractmethod
    def __xor__(self, other):
        """Define bitwise XOR for tensors."""
        lazy.compute(lazy.lazy(self).__xor__(other))

    @abstractmethod
    def __radd__(self, other):
        """Define right addition for tensors."""
        lazy.compute(lazy.lazy(self).__radd__(other))

    @abstractmethod
    def __rsub__(self, other):
        """Define right subtraction for tensors."""
        lazy.compute(lazy.lazy(self).__rsub__(other))

    @abstractmethod
    def __rmul__(self, other):
        """Define right multiplication for tensors."""
        lazy.compute(lazy.lazy(self).__rmul__(other))

    @abstractmethod
    def __rmatmul__(self, other):
        """Define right matrix multiplication for tensors."""
        lazy.compute(lazy.lazy(self).__rmatmul__(other))
    
    @abstractmethod
    def __rtruediv__(self, other):
        """Define right true division for tensors."""
        lazy.compute(lazy.lazy(self).__rtruediv__(other))

    @abstractmethod
    def __rfloordiv__(self, other):
        """Define right floor division for tensors."""
        lazy.compute(lazy.lazy(self).__rfloordiv__(other))

    @abstractmethod
    def __rmod__(self, other):
        """Define right modulo for tensors."""
        lazy.compute(lazy.lazy(self).__rmod__(other))
    
    @abstractmethod
    def __rdivmod__(self, other):
        """Define right division modulo for tensors."""
        lazy.compute(lazy.lazy(self).__rdivmod__(other))
    
    @abstractmethod
    def __rpow__(self, other):
        """Define right power for tensors."""
        lazy.compute(lazy.lazy(self).__rpow__(other))
    
    @abstractmethod
    def __rlshift__(self, other):
        """Define right left shift for tensors."""
        lazy.compute(lazy.lazy(self).__rlshift__(other))
    
    @abstractmethod
    def __rrshift__(self, other):
        """Define right right shift for tensors."""
        lazy.compute(lazy.lazy(self).__rrshift__(other))
    
    @abstractmethod
    def __rand__(self, other):
        """Define right bitwise AND for tensors."""
        lazy.compute(lazy.lazy(self).__rand__(other))
    
    @abstractmethod
    def __ror__(self, other):
        """Define right bitwise OR for tensors."""
        lazy.compute(lazy.lazy(self).__ror__(other))
    
    @abstractmethod
    def __rxor__(self, other):
        """Define right bitwise XOR for tensors."""
        lazy.compute(lazy.lazy(self).__rxor__(other))

    def __iadd__(self, other):
        """Define in-place addition for tensors."""
        lazy.compute(lazy.lazy(self).__iadd__(other))
    
    def __isub__(self, other):
        """Define in-place subtraction for tensors."""
        lazy.compute(lazy.lazy(self).__isub__(other))
    
    def __imul__(self, other):
        """Define in-place multiplication for tensors."""
        lazy.compute(lazy.lazy(self).__imul__(other))

    def __imatmul__(self, other):
        """Define in-place matrix multiplication for tensors."""
        lazy.compute(lazy.lazy(self).__imatmul__(other))

    def __itruediv__(self, other):
        """Define in-place true division for tensors."""
        lazy.compute(lazy.lazy(self).__itruediv__(other))

    def __ifloordiv__(self, other):
        """Define in-place floor division for tensors."""
        lazy.compute(lazy.lazy(self).__ifloordiv__(other))

    def __imod__(self, other):
        """Define in-place modulo for tensors."""
        lazy.compute(lazy.lazy(self).__imod__(other))
    
    def __idivmod__(self, other):
        """Define in-place division modulo for tensors."""
        lazy.compute(lazy.lazy(self).__idivmod__(other))

    def __ipow__(self, other):
        """Define in-place power for tensors."""
        lazy.compute(lazy.lazy(self).__ipow__(other))

    def __ilshift__(self, other):
        """Define in-place left shift for tensors."""
        lazy.compute(lazy.lazy(self).__ilshift__(other))
    
    def __irshift__(self, other):
        """Define in-place right shift for tensors."""
        lazy.compute(lazy.lazy(self).__irshift__(other))

    def __iand__(self, other):
        """Define in-place bitwise AND for tensors."""
        lazy.compute(lazy.lazy(self).__iand__(other))
    
    def __ior__(self, other):
        """Define in-place bitwise OR for tensors."""
        lazy.compute(lazy.lazy(self).__ior__(other))

    def __ixor__(self, other):
        """Define in-place bitwise XOR for tensors."""
        lazy.compute(lazy.lazy(self).__ixor__(other))

    def __neg__(self):
        """Define negation for tensors."""
        lazy.compute(lazy.lazy(self).__neg__())
    
    def __pos__(self):
        """Define unary plus for tensors."""
        lazy.compute(lazy.lazy(self).__pos__())
    
    def __abs__(self):
        """Define absolute value for tensors."""
        lazy.compute(lazy.lazy(self).__abs__())

    def __invert__(self):
        """Define bitwise NOT for tensors."""
        lazy.compute(lazy.lazy(self).__invert__())

    #TODO: should these raise an error? There are few others like that
    def __complex__(self):
        """Convert the tensor to a complex number."""
        pass  
    def __int__(self):
        """Convert the tensor to an integer."""
        pass
    def __float__(self):
        """Convert the tensor to a float."""
        pass


def permute_dims(arg, /, axis: Tuple[int, ...]):
    if isinstance(arg, lazy.LazyTensor):
        return lazy.permute_dims(arg, axis=axis)
    else:
        return lazy.compute(lazy.permute_dims(lazy.lazy(arg), axis=axis))
    
def expand_dims(
    x,
    /,
    axis: int | tuple[int, ...] = 0,
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.expand_dims(x, axis=axis)
    else:
        return lazy.compute(lazy.expand_dims(lazy.lazy(x), axis=axis))

def squeeze(
    x,
    /,
    axis: int | tuple[int, ...],
):
    if isinstance(x, lazy.LazyTensor):
        return lazy.squeeze(x, axis=axis)
    else:
        return lazy.compute(lazy.squeeze(lazy.lazy(x), axis=axis))

def reduce(
    op: Callable,
    x,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype = None,
    keepdims: bool = False,
    init = None):
    if isinstance(x, lazy.LazyTensor):
        return lazy.reduce(op, x, axis=axis, dtype=dtype, keepdims=keepdims, init=init)
    else:
        return lazy.compute(lazy.reduce(op, lazy.lazy(x), axis=axis, dtype=dtype, keepdims=keepdims, init=init))

def broadcast(f: Callable, src: LazyTensor, *args):
    if isinstance(src, lazy.LazyTensor):
        return lazy.broadcast(f, src, *args)
    else:
        return lazy.compute(lazy.broadcast(f, lazy.lazy(src), *args))

def prod(arr, /, axis=None):
    if isinstance(arr, lazy.LazyTensor):
        return lazy.prod(arr, axis=axis)
    else:
        return lazy.compute(lazy.prod(lazy.lazy(arr), axis=axis))

def multiply(x1, x2):
    if isinstance(x1, lazy.LazyTensor) and isinstance(x2, lazy.LazyTensor):
        return lazy.multiply(x1, x2)
    else:
        return lazy.compute(lazy.multiply(lazy.lazy(x1), lazy.lazy(x2)))


