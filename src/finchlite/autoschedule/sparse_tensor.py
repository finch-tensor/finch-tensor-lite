from typing import override
from finchlite.algebra import TensorFType
from finchlite.interface.eager import EagerTensor
import numpy as np

class SparseTensorFType(TensorFType):
    def __init__(self, shape: tuple, element_type: type):
        self.shape = shape
        self._element_type = element_type
    
    def __eq__(self, other):
        if not isinstance(other, SparseTensorFType):
            return False
        return self.shape == other.shape and self.element_type == other.element_type
    
    def __hash__(self):
        return hash((self.shape, self.element_type))

    @property
    def ndim(self):
        return len(self.shape)
    
    @property
    def shape_type(self):
        return self.shape
    
    @property
    def element_type(self):
        return self._element_type

    @property
    def fill_value(self):
        return 0

# currently implemented with COO tensor
class SparseTensor(EagerTensor):
    def __init__(self, data: np.array, coords: np.ndarray, shape: tuple, element_type=np.float64):
        self.coords = coords
        self.data = data
        self._shape = shape
        self._element_type = element_type

    # converts an eager tensor to a sparse tensor
    @classmethod
    def from_dense_tensor(cls, dense_tensor: np.ndarray):

        coords = np.where(dense_tensor != 0)
        data = dense_tensor[coords]
        shape = dense_tensor.shape
        element_type = dense_tensor.dtype.type  # Get the type, not the dtype
        # Convert coords from tuple of arrays to array of coordinates
        coords_array = np.array(coords).T
        return cls(data, coords_array, shape, element_type)

    @property
    def ftype(self):
        return SparseTensorFType(self.shape, self._element_type)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)
    
    # calculates the ratio of non-zero elements to the total number of elements
    @property
    def density(self):
        return self.coords.shape[0] / np.prod(self.shape)

    def __getitem__(self, idx: tuple):
        if len(idx) != self.ndim:
            raise ValueError(f"Index must have {self.ndim} dimensions")
        
        # coords is a 2D array where each row is a coordinate
        mask = np.all(self.coords == idx, axis=1)
        matching_indices = np.where(mask)[0]
        
        if len(matching_indices) > 0:
            return self.data[matching_indices[0]]
        return 0

    def to_dense(self) -> np.ndarray:
        dense_tensor = np.zeros(self.shape, dtype=self._element_type)
        for i in range(self.coords.shape[0]):
            dense_tensor[tuple(self.coords[i])] = self.data[i]
        return dense_tensor