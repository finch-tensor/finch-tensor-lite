# src/finch/tensor/michael_tensor.py
from typing import Any, override
import numpy as np
from finch.algebra import TensorFType
from finch.interface import EagerTensor
from finch.interface.eager import multiply, add

class MichaelTensorFType(TensorFType):
    """
    FType for MichaelTensor that describes its format/type metadata.
    """
    
    def __init__(self, dtype, ndim, special_property=None):
        self._dtype = dtype
        self._ndim = ndim
        self._special_property = special_property  # Your custom metadata
    
    def __eq__(self, other):
        if not isinstance(other, MichaelTensorFType):
            return False
        return (self._dtype == other._dtype and 
                self._ndim == other._ndim and
                self._special_property == other._special_property)
    
    def __hash__(self):
        return hash((self._dtype, self._ndim, self._special_property))
    
    def __repr__(self):
        return f"MichaelTensorFType(dtype={self._dtype}, ndim={self._ndim}, special={self._special_property})"
    
    @property
    def fill_value(self):
        """Default value for uninitialized elements"""
        # You can customize this based on your needs
        return np.zeros((), dtype=self._dtype)[()]
    
    @property
    def element_type(self):
        """Data type of tensor elements"""
        return self._dtype.type if hasattr(self._dtype, 'type') else self._dtype
    
    @property
    def shape_type(self):
        """Types of the shape dimensions"""
        # Usually int types for each dimension
        return tuple(np.int_ for _ in range(self._ndim))


class MichaelTensor(EagerTensor):
    """
    A custom tensor test implementation for me to play around with EagerTensor.
    """
    
    def __init__(self, data, special_property=1):
        # Convert to numpy array for simplicity, but you could use any storage
        self._data = np.asarray(data)
        self._special_property = special_property
    
    def __repr__(self):
        return f"MichaelTensor(shape={self.shape}, dtype={self._data.dtype}, special={self._special_property})"
    
    def __str__(self):
        return f"MichaelTensor:\n{self._data}"
    
    @property
    def shape(self):
        """Return the shape of the tensor"""
        return self._data.shape
    
    @property
    def ndim(self):
        """Return number of dimensions"""
        return self._data.ndim
    
    @property
    def dtype(self):
        """Return the data type"""
        return self._data.dtype
    
    @property
    def ftype(self):
        """Return the FType describing this tensor's format"""
        return MichaelTensorFType(
            dtype=self._data.dtype,
            ndim=self._data.ndim,
            special_property=self._special_property
        )
    
    def __getitem__(self, key):
        """Support indexing and slicing"""
        result = self._data[key]
        # If result is still an array, wrap it in MichaelTensor
        if isinstance(result, np.ndarray):
            return MichaelTensor(result, self._special_property)
        return result  # Return scalar as-is
    
    def asarray(self):
        """Convert to array format for Finch operations"""
        # This is what Finch uses to extract data for computation
        return self._data
    
    def to_numpy(self):
        """Convert to numpy array"""
        return self._data.copy()


    # override the __add__ method to change the behavior of the addition
    @override
    def __add__(self, other):
        return add(multiply(self, self._special_property), other)