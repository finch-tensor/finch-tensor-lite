from numpy.testing import assert_allclose, assert_equal


def finch_assert_equal(result, expected, **kwargs):
    if hasattr(result, "to_numpy"):
        result = result.to_numpy()
    if hasattr(expected, "to_numpy"):
        expected = expected.to_numpy()
    assert_equal(result, expected, **kwargs)


def finch_assert_allclose(result, expected, **kwargs):
    if hasattr(result, "to_numpy"):
        result = result.to_numpy()
    if hasattr(expected, "to_numpy"):
        expected = expected.to_numpy()
    assert_allclose(result, expected, **kwargs)
