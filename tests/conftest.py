import re

import pytest

from numpy.random import default_rng

from finch.finch_assembly import Module as AssemblyModule
from finch.finch_assembly.nodes import AssemblyPrinter
from finch.finch_logic import Field
from finch.finch_logic.nodes import LogicNode, LogicPrinter
from finch.finch_notation import Module as NotationModule
from finch.finch_notation.nodes import NotationPrinter
from finch.interface import get_default_scheduler, set_default_scheduler


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
            if isinstance(program, LogicNode):
                program = LogicPrinter()(program)
            elif isinstance(program, NotationModule):
                program = NotationPrinter()(program)
            elif isinstance(program, AssemblyModule):
                program = AssemblyPrinter()(program)
            else:
                raise TypeError(f"Unsupported program type: {type(program)}")

        # Apply additional test-specific substitutions
        if substitutions:
            for pattern_str, replacement in substitutions.items():
                compiled_pattern = re.compile(pattern_str)
                program = compiled_pattern.sub(replacement, program)

        file_regression.check(program, extension=extension, basename=basename)

    return _program_regression


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
