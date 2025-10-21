from importlib.abc import InspectLoader
from typing import Any, cast

import pytest

import numpy as np

import finchlite
from finchlite.autoschedule import (
    EinsumLowerer,
    InsumLowerer,
    optimize
)
import finchlite.finch_einsum as ein
import finchlite.finch_logic as logic
from finchlite.symbolic import gensym

@pytest.fixture
def rng():
    return np.random.default_rng(42)

def test_einsum_to_insum(plan: ein.Plan, bindings: dict[str, Any]):
    """Test converting an einsum plan to an insum plan"""
    lowerer = InsumLowerer()
    insum_plan, bindings = lowerer.optimize_plan(plan, bindings)

    interpreter = ein.EinsumInterpreter(bindings=bindings)
    result = interpreter(insum_plan)[0]
    result2 = interpreter(plan)[0]

    return np.allclose(result, result2)

def test_logic_to_insum(ir: logic.LogicNode):
    """Test converting a logic plan to an insum plan"""

    # Optimize into a plan
    var = logic.Alias(gensym("result"))
    plan = logic.Plan((logic.Query(var, ir), logic.Produces((var,))))
    optimized_plan = cast(logic.Plan, optimize(plan))

    # Lower to einsum IR
    lowerer = EinsumLowerer()
    einsum_plan, bindings = lowerer(optimized_plan)

    test_einsum_to_insum(einsum_plan, bindings)