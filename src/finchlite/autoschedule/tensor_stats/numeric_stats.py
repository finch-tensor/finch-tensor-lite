from abc import abstractmethod

from .tensor_stats import BaseTensorStats


class NumericStats(BaseTensorStats):
    @abstractmethod
    def estimate_non_fill_values(self) -> float:
        """
        Return an estimate on the number of non-fill values.
        """
        ...
