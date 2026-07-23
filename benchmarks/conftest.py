import pytest

import finch as ft
from finch.autoschedule import COMPILE_NUMBA, COMPILE_NUMBA_GALLEY


@pytest.fixture(
    params=[
        pytest.param(COMPILE_NUMBA_GALLEY, id="compile_numba_galley"),
        pytest.param(COMPILE_NUMBA, id="compile_numba"),
    ]
)
def scheduler(request):
    old = ft.get_default_scheduler()
    ft.set_default_scheduler(ctx=request.param)
    yield
    ft.set_default_scheduler(ctx=old)
