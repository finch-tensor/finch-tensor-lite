import re

import pytest

from numpy.random import default_rng

from finch.util.print import print_finch_program


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


# Regression fixture for compiler outputs
@pytest.fixture
def program_regression(file_regression, request):
    """
    Fixture for program and tree IR testing.
    """

    def _program_regression(
        program,
        extension=".txt",
        basename=None,
        substitutions: dict[str, str] | None = None,
    ):
        """
        Compares the program with the regression fixture.
        Args:
            program: The program to compare.
            extension: The file extension for the regression file.
            basename: Optional base name for the regression file.
                The final path will be:
                    `<test file parent>/<test file name>/<basename>.<extension>`
                If basename is None, the test name is used.
            substitutions: Optional dictionary of regex patterns and their replacements.
            E.g: {"<function (\\S+) at 0x[0-9a-fA-F]+>": "<function \\1 at 0x...>"}
        """
        if not isinstance(program, str):
            program = print_finch_program(program)

        # Apply additional test-specific substitutions
        if substitutions:
            for pattern_str, replacement in substitutions.items():
                compiled_pattern = re.compile(pattern_str)
                program = compiled_pattern.sub(replacement, program)

        file_regression.check(program, extension=extension, basename=basename)

    return _program_regression
