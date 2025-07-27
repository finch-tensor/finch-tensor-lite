import contextlib
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
# If you want to change the base directory, make sure to
# update pyproject.toml, and pre-commit config to make
# sure the references are not linted/formatted.
base_regr_dir = Path("tests/references")


@pytest.fixture
def lazy_datadir(request) -> Path:
    """
    Creates a unique per-test directory for reference files.
    E.g., tests/references/<module_name>/<test_name>/
    """
    # test_module will be something like "tests.test_logic_compiler"
    # We don't want the "tests." part
    module_parts = request.module.__name__.split(".")
    if module_parts[0] == "tests" and len(module_parts) > 1:
        test_module = "/".join(module_parts[1:])
    else:
        test_module = request.module.__name__.replace(".", os.sep)
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
    with contextlib.suppress(Exception):
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
                    with contextlib.suppress(OSError):
                        file.unlink()

    return (yield)


# a dictionary of regex pattern strings with their intended substitutions
substitution_rules = {r"<function (\S+) at 0x[0-9a-fA-F]+>": r"<function \1 at 0x...>"}
# compile into patterns
compiled_substitution_pairs = [
    (re.compile(pattern), replacement)
    for pattern, replacement in substitution_rules.items()
]


# Regression fixture for compiler outputs
@pytest.fixture
def program_regression(file_regression):
    """
    Fixture for program and tree regression testing.
    This overcomes the challenge of comparing non-deterministic outputs like
    memory addresses in program trees.
    """

    def _program_regression(
        program,
        formatter=pprint.pformat,
        extension=".txt",
        substitutions: dict[str, str] | None = None,
    ):
        """
        Compares the program with the regression fixture.
        Args:
            program: The program to compare.
            formatter: Optional formatter function to convert the AST to a string.
            extension: The file extension for the regression file.
            substitutions: Optional dictionary of regex patterns and their replacements.
            E.g: {"<function ( \\S+) at 0x[0-9a-fA-F]+>": "<function \\1 at 0x...>"}
        """
        if not isinstance(program, str):
            program = formatter(program)

        # Substitute matching regex patterns
        for pattern, replacement in compiled_substitution_pairs:
            program = pattern.sub(replacement, program)

        # If additional substitutions are provided, apply them
        # this allows us to add test-specific substitutions
        if substitutions:
            for pattern, replacement in substitutions.items():
                pattern = re.compile(pattern)
                program = pattern.sub(replacement, program)

        file_regression.check(program, extension=extension)

    return _program_regression
