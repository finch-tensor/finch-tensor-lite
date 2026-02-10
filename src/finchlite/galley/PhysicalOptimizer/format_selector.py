from enum import Enum
from typing import List


class LevelFormat(Enum):
    SPARSE_LIST = 1
    COO = 2
    DENSE = 3
    HASH = 4
    BYTEMAP = 5


def select_output_format(stats, loop_order: List[str], output_indices: List[str]) -> List[LevelFormat]:
    if not output_indices:
        return []
    
    formats = []
    
    for i in range(len(output_indices)):
        # gets the last i + 1 indices
        prefix = output_indices[-(i + 1):]
        
        sequential = all(idx in loop_order for idx in prefix)
        
        prev = 1.0 if len(prefix) == 1 else _get_nnz(stats, prefix[1:])
        curr = _get_nnz(stats, prefix)
        per_element = curr / prev if prev > 0 else 1.0
        
        size = _get_size(stats, prefix[0])
        density = per_element / size if size > 0 else 1.0
        
        mem = prev * size
        
        if density > 0.5 and mem < 3e10:
            formats.append(LevelFormat.DENSE)
        elif density > 0.05 and mem < 3e10:
            formats.append(LevelFormat.BYTEMAP)
        elif not sequential:
            formats.append(LevelFormat.HASH)
        else:
            formats.append(LevelFormat.SPARSE_LIST)
    
    return formats[::-1]


def _get_nnz(stats, indices: List[str]) -> float:
    return stats.estimate_nnz(indices) if hasattr(stats, 'estimate_nnz') else 1.0


def _get_size(stats, index: str) -> int:
    return stats.get_dim_size(index) if hasattr(stats, 'get_dim_size') else 1000
