
import pytest

import finchlite as fl
from finchlite.autoschedule import COMPILE_NUMBA, INTERPRET_NOTATION_GALLEY


@pytest.fixture(params=[
    pytest.param(INTERPRET_NOTATION_GALLEY, id="interpret_galley"),
    pytest.param(COMPILE_NUMBA, id="compile_numba"),
])
def scheduler(request):
    old = fl.get_default_scheduler()
    fl.set_default_scheduler(ctx=request.param)
    yield
    fl.set_default_scheduler(ctx=old)
