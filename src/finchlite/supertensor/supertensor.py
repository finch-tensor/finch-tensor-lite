import numpy as np
from typing import List, Tuple

class SuperTensor():
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
        logical_tns = np.empty(self.shape, dtype=self.base.dtype)
        for idx in np.ndindex(self.shape):
            logical_tns[idx] = self[idx]
        return f"SuperTensor(shape={self.shape}, base.shape={self.base.shape}, map={self.map})\n{logical_tns}"