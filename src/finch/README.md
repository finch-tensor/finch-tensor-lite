# Finch Tensor: Developer README

> **Note:** This README is AI-generated and may contain errors or omissions. If you find any inaccuracies or improvements, please update this file for the benefit of all contributors.

## Overview

This directory contains the implementation for Finch.py, a tensor compiler and runtime designed for extensibility and integration with other frameworks (such as NumPy). The codebase is organized to support both eager and lazy tensor computation, algebraic rewrites, and custom user extensions.

## Directory Structure

- `algebra/` — Algebraic properties, type promotion, and operator logic. Use this to define, register, and query custom properties for classes and operators.
- `interface/` — User-facing tensor APIs, including lazy and eager tensor implementations. The aim is to provide a consistent interface for tensor operations according to the [array API specification](https://data-apis.org/array-api/latest/).
- `symbolic/` — Symbolic utilities for tensor computation.
- `finch_logic/` — Internal logic node representations and manipulation.
- `overrides.py` — Mechanisms for user and library overrides.
- `__init__.py` — Module initialization and exports.

## Contributing

- **If you find an error or unclear section in this README or the code, please fix it or open an issue.**
- Follow the code style and conventions used throughout the project.
- Add or update docstrings for all public functions and classes.
- To to write thorough tests for new features and bug fixes in the `tests/` directory.

## Assertions and Validation

- **Do not use `assert` statements for user-facing validation.**
    - Relying on `assert` for input validation or error checking is unsafe, as all `assert` statements are removed when Python is run with the `-O` (optimize) flag, which may be used for production builds.
    - Use explicit error handling (e.g., `if ...: raise ValueError(...)`) for all user-facing checks and input validation. For user facing APIs, you should raise exceptions according to the [array API specification](https://data-apis.org/array-api/latest/).
- `assert` statements **may** be used for internal debugging, invariants, and sanity checks that are not critical for production correctness.

## Testing

- Tests are located in the `tests/` directory at the project root.
- Use `poetry run pytest` to run the test suite.
- Add tests for all new features and bug fixes.

## Documentation

- All public APIs should have clear docstrings following the NumPy style.
- Update this README as the codebase evolves.

---

*This file is mostly AI-generated. Please help improve it as you work on the project!*
