# AGENTS instruction

## Coding style

- Don't explain every line with a separate comment, use comments for complex chunks of code,
  or DSL logic. Don't write docstrings for single-line or simple functions/methods.
- Use structures and code style that are already present in the codebase. Don't introduce
  another approach for new changes. For example, use structural pattern matching instead of
  `isinstance()` for dataclasses.
- Read `CONTRIBUTING.md` file for more details how to build the project and run tests.
