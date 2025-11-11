import numpy as np
from typing import List, Tuple

class SuperTensor():
    """
    Represents a tensor using a base tensor of lower order.

    Attributes:
        shape: `Tuple[int, ...]`
            The logical shape of the tensor.
        base: `np.ndarray`
            The base tensor.
        map: `List[List[int]]`
            Maps each mode of the base tensor to a ordered list of the logical modes which are flattened into the base mode.
            The ordering of each list defines the order in which the logical modes are flattened.

            Example: map = [[0, 2], [3], [4, 1]] indicates that the base tensor has three modes and the logical tensor has five modes.
                - Base mode 0 corresponds to logical modes 0 and 2.
                - Base mode 1 corresponds to logical mode 3.
                - Base mode 2 corresponds to logical modes 4 and 1.
    """

    shape: Tuple[int, ...]
    base: np.ndarray
    map: List[List[int]]

    def __init__(self, shape: Tuple[int, ...], base: np.ndarray, map: List[List[int]]):
        self.shape = shape
        self.base = base
        self.map = map
    
    @property
    def N(self) -> int:
        return len(self.shape)
    
    @property
    def B(self) -> int:
        return self.base.ndim
    
    @classmethod
    def from_logical(cls, tns: np.ndarray, map: List[List[int]]):
        """
        Constructs a SuperTensor from a logical tensor and a mode map.

        Args:
            tns: `np.ndarray`
                The logical tensor.
            map: `List[List[int]]`
                The mode map.
        """
        shape = tns.shape

        base_shape = [0] * len(map)
        for b, logical_idx_group in enumerate(map):
            dims = [shape[m] for m in logical_idx_group]
            base_shape[b] = np.prod(dims) if dims else 1

        perm = [i for logical_idx_group in map for i in logical_idx_group]
        permuted_tns = np.transpose(tns, perm)
        base = permuted_tns.reshape(tuple(base_shape))

        return SuperTensor(shape, base, map)

    def __getitem__(self, coords: Tuple[int, ...]):
        """
        Accesses an element of the SuperTensor using logical coordinates.

        Args:
            coords: `Tuple[int, ...]`
                The logical coordinates to access.

        Returns:
            The value in the SuperTensor at the given logical coordinates.

        Raises:
            IndexError: The number of input coordinates does not match the order of the logical tensor.
        """

        if len(coords) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)} indices, got {len(coords)} indices")
        
        base_coords = [0] * self.B
        for b, logical_modes in enumerate(self.map):
            if len(logical_modes) == 1:
                base_coords[b] = coords[logical_modes[0]]
            else:
                subshape = tuple(self.shape[m] for m in logical_modes)
                subidcs = tuple(coords[m] for m in logical_modes)                
                linear_idx = 0
                for dim, idx in zip(subshape, subidcs):
                    linear_idx = linear_idx * dim + idx
                base_coords[b] = linear_idx

        return self.base[tuple(base_coords)]
    
    def __repr__(self):
        """
        Returns a string representation of the SuperTensor.

        Includes the logical shape, the base shape, the mode map, and the logical tensor itself.

        Returns:
            A string representation of the SuperTensor.
        """
        logical_tns = np.empty(self.shape, dtype=self.base.dtype)
        for idx in np.ndindex(self.shape):
            logical_tns[idx] = self[idx]
        return f"SuperTensor(shape={self.shape}, base.shape={self.base.shape}, map={self.map})\n{logical_tns}"