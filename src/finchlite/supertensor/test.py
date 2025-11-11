import numpy as np
from finchlite import finch_einsum as ein
from finchlite import supertensor as stns

# ========= TEST SuperTensorEinsumInterpreter =========
# Very simple test case

A = np.random.randint(1,5,(3,2,4))
supertensor_A = stns.SuperTensor.from_logical(A, [[0],[1,2]])

B = np.random.randint(1,5,(2,4,5))
supertensor_B = stns.SuperTensor.from_logical(B, [[0,1],[2]])

print(f"SuperTensor A:\n{supertensor_A}\n")
print(f"SuperTensor B:\n{supertensor_B}\n")

# Using regular EinsumInterpreter
einsum_AST, bindings = ein.parse_einsum("ikl,klj->ij", A, B)
interpreter = ein.EinsumInterpreter(bindings=bindings)
output = interpreter(einsum_AST)  
result = bindings[output[0]]
print(f"Regular einsum interpreter result:\n{result}\n")

# Using SuperTensorEinsumInterpreter
supertensor_einsum_AST, supertensor_bindings = ein.parse_einsum("ikl,klj->ij", supertensor_A, supertensor_B)

# print(f"SuperTensor einsum AST info:\n")
# print(f"{supertensor_einsum_AST}\n")
# print(f"{supertensor_bindings}\n")

supertensor_interpreter = stns.SuperTensorEinsumInterpreter(bindings=supertensor_bindings)
output = supertensor_interpreter(supertensor_einsum_AST)
result = supertensor_bindings[output[0]]
print(f"SuperTensor einsum interpreter result:\n{result}")

# TEST SuperTensor class
# A = np.random.randint(1,5,(2,3,4))
# print(A)
# supertensor = stns.SuperTensor.from_logical(A, [[0,1],[2]])
# print(supertensor)
# print(supertensor.base)

# TEST _group_indices()
# groups = stns.SuperTensorEinsumInterpreter._group_indices("out", ["p", "i", "j"], [("A", ["p", "q", "i", "k"]), ("B", ["p", "r", "k", "j"])])
# print(groups)

# TEST _collect_accesses()
# accesses = stns.SuperTensorEinsumInterpreter._collect_accesses(einsum_AST)
# print(f"\nAccesses:\n{accesses}")