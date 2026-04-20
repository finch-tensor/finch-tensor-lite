from abc import abstractmethod
import numpy as np
from .tensor_stats import BaseTensorStats


class NumericStats(BaseTensorStats):
    @abstractmethod
    def estimate_non_fill_values(self) -> float:
        """
        Return an estimate on the number of non-fill values.
        """
        ...
    
    @abstractmethod
    def get_embedding(self) -> np.ndarray:
        """
        Returns vector embedding for the stat.
        """
        ...