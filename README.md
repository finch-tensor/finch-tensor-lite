# finch-tensor-lite

A pure-Python rewrite of [Finch.jl](https://github.com/finch-tensor/Finch.jl), designed to eventually replace [finch-tensor](https://pypi.org/project/finch-tensor/).

## Features

- **Write Once, Run Efficiently:** Express tensor computations in a clear, NumPy-like style—`finch-tensor-lite` automatically optimizes and fuses operations for both dense and sparse data.
- **Flexible Execution:** Effortlessly switch between eager (immediate) and lazy (deferred, fused) evaluation to balance simplicity and performance.
- **Advanced Optimization:** Benefit from algebraic and symbolic rewrites, constant folding, and other compiler-level optimizations—no manual tuning required.
- **Rich Array Support:** Work seamlessly with dense, sparse, and structured arrays, including custom background (zero) values and user-defined formats.
- **Interoperable:** Integrates smoothly with NumPy and the broader Python scientific ecosystem.
- **Modern API:** Follows the [Array API specification](https://data-apis.org/array-api/latest/) for consistency and compatibility.

## Installation

Install from PyPI:
```bash
pip install finch-tensor-lite
```

For development (with tests and pre-commit hooks):
```bash
git clone https://github.com/finch-tensor/finch-tensor-lite.git
cd finch-tensor-lite
poetry install --extras test
poetry run pre-commit install
```

## Usage

```python
import finch_tensor_lite as finch
# Example usage here
```

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, development setup, and best practices.
