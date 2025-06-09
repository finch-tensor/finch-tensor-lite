# Finch Tensor: Contributing Guide

Thank you for your interest in contributing! Please read the following guidelines to help us maintain a high-quality, collaborative codebase.

## Code of Conduct

We adhere to the [Python Code of Conduct](https://policies.python.org/python.org/code-of-conduct/).

## Collaboration Practices

- New to contributing? Welcome! See GitHub’s [pull request guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).
- Please follow the [SciML Collaborative Practices](https://docs.sciml.ai/ColPrac/stable/) and [GitHub Collaborative Practices](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes).
- Use the convention `<initials>/<branch-name>` for pull request branches (e.g., `ms/scheduler-pass`). This makes branch management easier.

## Packaging

We use [poetry](https://python-poetry.org/) for packaging.

To install for development:
```bash
poetry install --extras test
```
This installs the project and development dependencies.

## Publishing

Publishing to PyPI is handled by a manual GitHub Action workflow using Poetry. The workflow uses the version in `pyproject.toml` and automates tagging and GitHub releases.

**Before publishing:**
- Update the version in `pyproject.toml` following semantic versioning.

**To publish:**
- Manually trigger the "Publish" action from the GitHub Actions tab after updating the version.

**Notes:**
- If the version is not updated, publishing will fail.
- Check action logs for completion or errors.

## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) for formatting, linting, and typing checks.

To install hooks:
```bash
poetry run pre-commit install
```
To run hooks manually:
```bash
poetry run pre-commit run -a
```

## Testing

We use [pytest](https://docs.pytest.org/en/latest/) for testing.

To run tests:
```bash
poetry run pytest
```

### Static Type Checking

Type checks are run with mypy as part of pytest, but you can run it manually:
```bash
poetry run mypy ./src/
```

## Codebase Overview

### Directory Structure
`src/finch/`:

- `algebra/` — Core algebraic logic: defines algebraic properties, type promotion, operator logic, and utilities for querying and registering algebraic properties.
- `autoschedule/` — Automatic scheduling and optimization passes for tensor computations, including logic graph transformations and compiler passes.
- `codegen/` — Code generation backends, including C and NumPy buffer support, for compiling and running tensor computations.
- `finch_assembly/` — Intermediate representation for low-level tensor operations, including assembly nodes, interpreters, and buffer abstractions.
- `finch_logic/` — Internal logic node representations for symbolic tensor computation, including logic trees, expressions, and interpreters.
- `interface/` — User-facing tensor APIs, including lazy and eager tensor implementations, array API overrides, and computation fusion.
- `symbolic/` — Symbolic utilities for manipulating computation graphs, including term rewriting, symbolic environments, and unique name generation.
- `util/` — Utility modules for configuration, caching, and other shared infrastructure.
- `interface/overrides.py` — Mechanisms for user and library overrides, including NumPy ufunc compatibility and array API support.
- `__init__.py` — Module initialization and exports for the Finch package.

## Assertions and Validation

- **Do not use `assert` statements for user-facing validation.**
    - `assert` statements are removed when Python is run with the `-O` (optimize) flag.
    - Use explicit error handling (e.g., `if ...: raise ValueError(...)`) for all user-facing checks, following the [array API specification](https://data-apis.org/array-api/latest/).
- `assert` statements may be used for internal debugging, invariants, and sanity checks that are not critical for production correctness.

## Testing

- Tests are located in the `tests/` directory at the project root.
- Add tests for all new features and bug fixes. Write thorough tests for new features and bug fixes in the `tests/` directory.

## Documentation

- All public APIs should have clear docstrings following the NumPy style.
- Update this file as the codebase evolves.

---
**If you find an error or unclear section, please fix it or open an issue.**
