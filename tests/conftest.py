import os
import pprint
import re
from pathlib import Path

import pytest

from numpy.random import default_rng


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


# Regression fixtures, helpers, and hooks
base_regr_dir = Path("tests/regressions")


@pytest.fixture
def lazy_datadir(request) -> Path:
    """
    Creates a unique per-test directory for regression files.
    E.g., tests/regressions/<module_name>/<test_name>/
    """
    # Derive names
    # test_module will be something like "tests.test_logic_compiler"
    # We don't want the "tests." part
    test_module = os.path.join(*(request.module.__name__.split(".")[1:]))
    test_name = request.node.name

    # Construct path
    test_dir = base_regr_dir / test_module / test_name

    # Ensure it exists
    test_dir.mkdir(parents=True, exist_ok=True)

    return test_dir


@pytest.fixture
def original_datadir(lazy_datadir) -> Path:
    return lazy_datadir


@pytest.fixture
def preserve_obtained(request):
    """
    Fixture to preserve obtained files after the test run.
    """
    request.node.preserve_obtained = True


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to clean up obtained files. Obtained files are generated
    in regression tests to compare against the original file.
    ### This hook ensures that:
    - If the test passes, and preserve_obtained is not set to True,
        remove all `*.obtained*` files in the test directory.
    - If the test fails, do not remove any files. A diff is generated from
    the html report.
    """
    if (
        call.when == "call"  # Only clean up after the test run
        and call.excinfo is None  # Only clean up if the test passed
        and not (
            hasattr(item, "preserve_obtained") and item.preserve_obtained
        )  # Only clean up if preserve_obtained is not set
    ):
        # Test passed clean up obtained files
        test_module = Path(item.location[0].replace(".py", "")).parts[1:]
        test_module = os.path.join(*test_module)
        test_name = item.name
        test_dir: Path = base_regr_dir / test_module / test_name
        if test_dir.exists():
            # find and remove all .obtained files in the directory
            for file in test_dir.glob("*.obtained*"):
                file.unlink()
    return (yield)


# Regression fixture for compiler outputs
@pytest.fixture
def program_regression(file_regression, tmp_path):
    """
    Fixture for program and tree regression testing.
    This overcomes the challenge of comparing non-deterministic outputs like
    memory addresses in program trees.
    """

    def _program_regression(program, formatter=pprint.pformat, extension=".txt"):
        """
        Compares the AST with the regression fixture.
        Args:
            ast: The AST to compare.
            formatter: Optional formatter function to convert the AST to a string.
        """
        if not isinstance(program, str):
            program = formatter(program)
        # Replace memory addresses with ...
        file_regression.check(re.sub(r"0x\w+", "...", program), extension=extension)

    return _program_regression
