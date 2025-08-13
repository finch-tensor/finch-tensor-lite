from pathlib import Path

import pytest

from numpy.random import default_rng

from finch.finch_logic import Field
from finch.interface import get_default_scheduler, set_default_scheduler


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
    yield set_default_scheduler(interpret_logic=True)
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
