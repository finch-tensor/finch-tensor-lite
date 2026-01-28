import re
from pathlib import Path

import pytest

from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_equal

from finchlite import get_default_scheduler, set_default_scheduler
from finchlite.finch_logic import Field
from finchlite.interface.fuse import COMPILE_NUMBA, INTERPRET_LOGIC


@pytest.fixture(scope="session")
def lazy_datadir() -> Path:
    return Path(__file__).parent / "reference"


@pytest.fixture(scope="session")
def original_datadir() -> Path:
    return Path(__file__).parent / "reference"


@pytest.fixture
def rng():
    return default_rng(42)


@pytest.fixture
def random_wrapper(rng):
    def _random_wrapper_applier(arrays, wrapper):
        """
        Applies a wrapper to each array in the list with a 50% chance.
        Args:
            arrays: A list of NumPy arrays.
            wrapper: A function to apply to the arrays.
        """
        return [wrapper(arr) if rng.random() > 0.5 else arr for arr in arrays]

    return _random_wrapper_applier


@pytest.fixture
def interpreter_scheduler():
    ctx = get_default_scheduler()
    yield set_default_scheduler(ctx=INTERPRET_LOGIC)
    set_default_scheduler(ctx=ctx)


@pytest.fixture
def numba_compiler():
    ctx = get_default_scheduler()
    yield set_default_scheduler(ctx=COMPILE_NUMBA)
    set_default_scheduler(ctx=ctx)


@pytest.fixture
def tp_0():
    return (Field("A1"), Field("A3"))


@pytest.fixture
def tp_1():
    return (Field("A0"), Field("A1"), Field("A2"), Field("A3"))


@pytest.fixture
def tp_2():
    return (Field("A3"), Field("A1"))


@pytest.fixture
def tp_3():
    return (Field("A0"), Field("A3"), Field("A2"), Field("A1"))


# TODO Remove once we solve #280
def reset_name_counts(prgm: str) -> str:
    counter = 0
    visited = set()
    matched = re.finditer(r"#(?P<alias>\w+)#(?P<counter>\d+)", prgm)
    for m in matched:
        label = m.group(0)
        if label not in visited:
            visited.add(label)
            counter += 1
            prgm = prgm.replace(label, f"#{m.group('alias')}#{counter}")
    return prgm


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
