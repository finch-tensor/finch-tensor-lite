import numpy as np

from finchlite import finch_einsum as ein
from finchlite import supertensor as stns

# ========= TEST SuperTensorEinsumInterpreter =========
# Very simple test case

rng = np.random.default_rng()

A = rng.integers(1, 5, (3, 2, 4))
supertensor_A = stns.SuperTensor.from_logical(A, [[0], [1, 2]])

B = rng.integers(1, 5, (2, 5, 4, 3))
supertensor_B = stns.SuperTensor.from_logical(B, [[3], [1], [2, 0]])

print(f"SuperTensor A:\n{supertensor_A}\n")
print(f"SuperTensor B:\n{supertensor_B}\n")

# Using regular EinsumInterpreter
einsum_AST, bindings = ein.parse_einsum("ikl,kjln->ijn", A, B)
print("Bindings", bindings)

interpreter = ein.EinsumInterpreter()
output = interpreter(einsum_AST, bindings=bindings)
result = output[0]
print(f"Regular einsum interpreter result:\n{result}\n")

# Using SuperTensorEinsumInterpreter
supertensor_einsum_AST, supertensor_bindings = ein.parse_einsum(
    "ikl,kjln->ijn", supertensor_A, supertensor_B
)

# print(f"SuperTensor einsum AST info:\n")
# print(f"{supertensor_einsum_AST}\n")
# print(f"{supertensor_bindings}\n")

supertensor_interpreter = stns.SuperTensorEinsumInterpreter(
    bindings=supertensor_bindings
)
output = supertensor_interpreter(supertensor_einsum_AST)
result = output[0]
print(f"SuperTensor einsum interpreter result:\n{result}")
