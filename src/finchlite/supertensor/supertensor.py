import numpy as np

class SuperTensor():
    """
    Represents a tensor using a base tensor of lower order.

    Attributes:
        shape: `tuple[int, ...]`
            The logical shape of the tensor.
        base: `np.ndarray`
            The base tensor.
        mode_map: `list[list[int]]`
            Maps each mode of the base tensor to a ordered list of the logical modes which are flattened into the base mode.
            The ordering of each list defines the order in which the logical modes are flattened.

            Example: mode_map = [[0, 2], [3], [4, 1]] indicates that the base tensor has three modes and the logical tensor has five modes.
                - Base mode 0 corresponds to logical modes 0 and 2.
                - Base mode 1 corresponds to logical mode 3.
                - Base mode 2 corresponds to logical modes 4 and 1.
    """

    shape: tuple[int, ...]
    base: np.ndarray
    mode_map: list[list[int]]

    def __init__(self, shape: tuple[int, ...], base: np.ndarray, mode_map: list[list[int]]):
        self.shape = shape
        self.base = base
        self.mode_map = mode_map
    
    @property
    def N(self) -> int:
        return len(self.shape)
    
    @property
    def B(self) -> int:
        return self.base.ndim
    
    @classmethod
    def from_logical(cls, tns: np.ndarray, mode_map: list[list[int]]):
        """
        Constructs a SuperTensor from a logical tensor and a mode map.

        Args:
            tns: `np.ndarray`
                The logical tensor.
            mode_map: `list[list[int]]`
                The mode map.
        """
        shape = tns.shape

        base_shape = [0] * len(mode_map)
        for b, logical_idx_group in enumerate(mode_map):
            dims = [shape[m] for m in logical_idx_group]
            base_shape[b] = np.prod(dims) if dims else 1

        perm = [i for logical_idx_group in mode_map for i in logical_idx_group]
        permuted_tns = np.transpose(tns, perm)
        base = permuted_tns.reshape(tuple(base_shape))

        return SuperTensor(shape, base, mode_map)

    def __getitem__(self, coords: tuple[int, ...]):
        """
        Accesses an element of the SuperTensor using logical coordinates.

        Args:
            coords: `tuple[int, ...]`
                The logical coordinates to access.

        Returns:
            The value in the SuperTensor at the given logical coordinates.

        Raises:
            IndexError: The number of input coordinates does not match the order of the logical tensor.
        """

        if len(coords) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)} indices, got {len(coords)} indices")
        
        base_coords = [0] * self.B
        for b, logical_modes in enumerate(self.mode_map):
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
        return f"SuperTensor(shape={self.shape}, base.shape={self.base.shape}, mode_map={self.mode_map})\n{logical_tns}"