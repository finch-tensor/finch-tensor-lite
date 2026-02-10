from finchlite.galley.PhysicalOptimizer import LevelFormat, select_output_format


class Stats:
    """
    Helper class to store the statistics of a tensor. Just for test purposes.
    """
    def __init__(self, nnz_per_dim, dim_size):
        """
        Initialize the statistics of a tensor.

        Args:
            nnz_per_dim: The number of nonzeroes per dimension.
            dim_size: The size of the dimension.
        """
        self.nnz_per_dim = nnz_per_dim
        self.dim_size = dim_size
    
    def estimate_nnz(self, indices):
        """
        Estimate the number of nonzeroes for a given set of indices.
        """
        return self.nnz_per_dim ** len(indices) if indices else 1.0
    
    def get_dim_size(self, index):
        return self.dim_size


def test_dense():
    """
    800 nonzeroes, 1K dimensions. Should be dense.
    """
    stats = Stats(800.0, 1000)
    formats = select_output_format(stats, ['i', 'j'], ['i', 'j'])
    assert formats[0] == LevelFormat.DENSE


def test_sparse():
    """
    Only 10 nonzeroes, 10K dimensions. Should be sparse.
    """
    stats = Stats(10.0, 10000)
    formats = select_output_format(stats, ['i', 'j'], ['i', 'j'])
    assert formats[0] == LevelFormat.SPARSE_LIST


def test_empty():
    """
    No nonzeroes, no dimensions. Should be empty.
    """
    stats = Stats(100.0, 1000)
    formats = select_output_format(stats, [], [])
    assert formats == []
