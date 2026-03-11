# Finch Tensor: Contributing Guide

Thank you for your interest in contributing! Please read the following guidelines
to help us maintain a high-quality, collaborative codebase.

## Code of Conduct

We adhere to the [Python Code of Conduct](https://policies.python.org/python.org/code-of-conduct/).

## Collaboration Practices

For those who are new to the process of contributing code, welcome! We value your
contribution, and are excited to work with you. GitHub's [pull request guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
will walk you through how to file a PR.

Please follow the [SciML Collaborative Practices](https://docs.sciml.ai/ColPrac/stable/)
and [Github Collaborative Practices](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes)
guides to help make your PR easier to review.

In this repo, please use the convention <initials>/<branch-name> for pull request
branch names, e.g. ms/scheduler-pass. This way in bash when you type your initials
git checkout ms/ and <tab> you can see all your branches. We will use other names
for special purposes.

### Developer Interface

Finch uses [Pixi](https://pixi.prefix.dev) for development.

See below for how to perform development tasks via `pixi run`, which will install Finch and its development dependencies as needed.

### Publishing

The "Publish" GitHub Action is a manual workflow for publishing Python packages to
PyPI using Poetry. It handles the version management based on the `pyproject.toml`
file and automates tagging and creating GitHub releases.

#### Version Update

Before initiating the "Publish" action, update the package's version number in `pyproject.toml`.
Follow semantic versioning guidelines for this update.

#### Triggering the Action

The action is triggered manually. Once the version in `pyproject.toml` is updated,
manually start the "Publish" action from the GitHub repository's Actions tab.

#### Process and Outcomes

On successful execution, the action publishes the package to PyPI and tags
the release in the GitHub repository. If the version number is not updated,
the action fails to publish to PyPI, and no tagging or release is done. In
case of failure, correct the version number and rerun the action.

#### Best Practices

- Ensure the version number in `pyproject.toml` is updated before triggering the action.
- Regularly check action logs for successful completion or to identify issues.

### Pre-commit hooks

Pull requests must pass some formatting, linting, and typing checks before we can
merge them. These checks can be run automatically before you make commits, which
is why they are sometimes called "pre-commit hooks". We use [pre-commit](https://pre-commit.com/)
to run these checks.

To install pre-commit hooks to run before committing, run:
```bash
pixi run pre-commit-install
```
If you prefer to instead run pre-commit hooks manually, run:
```bash
pixi run pre-commit
```

### Testing

Finch uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run the
tests:

```bash
pixi run test
# or to test with the mininum supported Python version:
pixi run --environment=mindeps test
```

- Tests are located in the `tests/` directory at the project root.
- Write thorough tests for your new features and bug fixes.

#### Optional Static Type Checking

The pytest will run mypy to check for type errors, so you shouldn't need to run it manually.
In case you do need to run mypy manually, you can do so with:

```bash
pixi run type-check
```

#### Regression Tests

`pytest-regression` is used to ensure that compiler outputs remain consistent across
changes, and to better understand the impacts of compiler changes on the test outputs.
To regenerate regression test outputs, run pytest with the `--regen-all` flag. Those
who are curious can consult the [`pytest-regression` docs](https://pytest-regressions.readthedocs.io/en/latest/overview.html#using-data-regression).

## Development & Code Style

### Finch Assembly VS Code extension

When your Finch Assembly code is becoming unintelligible due to size and verbosity
of assembly nodes we recommend using Finch Assembly parser for Python's multiline
strings. It supports a custom language for constructing assembly nodes in a more
concise manner.

```py
code = """finch
    for (i in 0 : last_idx - 1)
        arr[i + 1] += arr[i]
    end
"""

asm_node = parse_assembly(expr, vars)
```

There is a dedicated Finch Assembly VS Code extension for highlighting any multiline
string starting with a `finch` tag. The extension is not yet available on VS Code marketplace.
You can download it here: [vscode-finch-assembly](https://github.com/finch-tensor/vscode-finch-assembly/releases).

### Assertions and Validation

- **Do not use `assert` statements for user-facing validation.**
    - `assert` statements are removed when Python is run with the `-O` (optimize) flag.
    - Use explicit error handling (e.g., `if ...: raise ValueError(...)`) for all
      user-facing functions, following the [array API specification](https://data-apis.org/array-api/latest/).
    - User-facing functions are anything exposed from `__all__` in the toplevel `__init__.py`.
- `assert` statements may be used for internal debugging, invariants, and sanity checks
  that are not critical to production behavior.

### Getters and Setters
- Use `@property` decorators for getters and setters.
- This means you may need to define a private `_foo` attribute in your dataclass to
  implement the `foo` property.
- Avoid using `get_` and `set_` prefixes in method names.
- `get_` and `set_` prefixes are allowed for global getters and setters, such as
  `util.get_version()`.

### Path Handling
- **Use `pathlib.Path` instead of `os.path` for file and directory operations.**
- `pathlib.Path` provides a more modern interface for path manipulation.
- It's cross-platform by default and more readable than string-based `os.path` operations.
- Example: Use `Path("dir") / "file.txt"` instead of `os.path.join("dir", "file.txt")`.

### Compiler Passes
- In general, each compiler pass should be implemented as a separate callable class,
  inheriting from some subclass of `finchlite.symbolic.Stage`.
- Each pass should have a clear, single, documented responsibility.
- Try to separate passes that do different things into different classes, rather than
  building an all-in-one monolithic pass.
- Files which involve more than one IR should not import the AST nodes of one IR into
  the file directly. For clarity, nodes of both IRs should be referred to with
  qualified names, e.g. `lgc.Plan` and `ntn.Loop`.

---

**If you find an error or unclear section, please fix it or open an issue.**
