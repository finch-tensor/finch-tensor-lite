# finch-tensor-lite

A pure-Python rewrite of [Finch.jl](https://github.com/finch-tensor/Finch.jl), designed to eventually replace [finch-tensor](https://pypi.org/project/finch-tensor/).

## Source
The source code for `finch-tensor-lite` is available on GitHub at [https://github.com/finch-tensor/finch-tensor-lite](https://github.com/FinchTensor/finch-tensor-lite)

## Installation

`finch-tensor-lite` is available on PyPi, and can be installed with pip:
```bash
pip install finch-tensor-lite
```

## Usage

```python
import finch
import numpy as np

a = np.arange(...).reshape(...)  # A large NumPy array
b = np.arange(...).reshape(...)  # Another large NumPy array
c = finch.defer(a) @ finch.defer(b)  # Deferred tensor operation
c = c + np.ones_like(c)  # Another operation on the deferred tensor
# c is still deferred; no computation yet
finch.compute(c)  # Trigger computation
```

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, development setup, and best practices.
