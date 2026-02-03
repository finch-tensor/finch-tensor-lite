import numpy as np

from finchlite.algebra import TensorFType
from finchlite.interface.eager import EagerTensor


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

    def __call__(self, shape: tuple) -> "SparseTensor":
        """Create an empty SparseTensor with the given shape."""
        data: np.typing.NDArray = np.array([], dtype=self._element_type)
        coords: np.typing.NDArray = np.empty((len(shape), 0), dtype=np.intp)
        return SparseTensor(data, coords, shape, self._element_type)

    # converts an eager tensor to a sparse tensor
    @classmethod
    def from_numpy(cls, dense_tensor: np.ndarray):
        coords = np.where(dense_tensor != 0)
        data = dense_tensor[coords]
        shape = dense_tensor.shape
        element_type = dense_tensor.dtype.type
        coords_array = np.array(coords)
        return SparseTensor(data, coords_array, shape, element_type)


# currently implemented with COO tensor
class SparseTensor(EagerTensor):
    def __init__(
        self,
        elems: np.typing.NDArray,
        coords: np.typing.NDArray,
        shape: tuple,
        element_type=np.float64,
    ):
        assert len(elems.shape) == 1
        assert len(coords.shape) == 2
        assert coords.shape[0] == len(shape)
        assert coords.shape[1] == len(elems)

        self._elems = elems
        self._coords = coords
        self._shape = shape
        self._element_type = element_type

    @property
    def ftype(self):
        return SparseTensorFType(self.shape, self._element_type)

    @property
    def shape(self):
        return self._shape

    @property
    def coords(self):
        return self._coords

    @property
    def elems(self):
        return self._elems

    @property
    def ndim(self) -> np.intp:
        return np.intp(len(self._shape))

    # calculates the ratio of non-zero elements to the total number of elements
    @property
    def density(self):
        return len(self._elems) / np.prod(self._shape)

    def __getitem__(self, idx: tuple):
        if len(idx) != self.ndim:
            raise ValueError(f"Index must have {self.ndim} dimensions")

        # coords is a 2D array where each row is a coordinate
        mask = np.all(self.coords == idx, axis=0)
        matching_indices = np.where(mask)[0]

        if len(matching_indices) > 0:
            return self.elems[matching_indices[0]]
        return 0

    def __str__(self):
        return (
            f"SparseTensor(data={self.elems}, coords={self.coords},"
            f" shape={self.shape}, element_type={self._element_type})"
        )

    def to_dense(self) -> np.ndarray:
        dense_tensor = np.zeros(self.shape, dtype=self._element_type)
        for i in range(self.coords.shape[1]):
            dense_tensor[tuple(self.coords[:, i])] = self._elems[i]
        return dense_tensor
