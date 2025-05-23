# finch-tensor-lite

The pure-python rewrite of [Finch.jl](https://github.com/finch-tensor/Finch.jl) to eventually replace [finch-tensor](https://pypi.org/project/finch-tensor/)

## Source

The source code for `finch-tensor-lite` is available on GitHub at [https://github.com/finch-tensor/finch-tensor-lite](https://github.com/FinchTensor/finch-tensor-lite)

## Installation

`finch-tensor-lite` is available on PyPi, and can be installed with pip:
```bash
pip install finch-tensor-lite
```

## Contributing

### Code Of Conduct

We adhere to the [Python Code of Conduct](https://policies.python.org/python.org/code-of-conduct/)

### Collaboration Practices

For those who are new to the process of contributing code, welcome! We value your contribution, and are excited to work with you. GitHub's [pull request guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) will walk you through how to file a PR.

Please follow the [SciML Collaborative Practices](https://docs.sciml.ai/ColPrac/stable/) and [Github Collaborative Practices](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes) guides to help make your PR easier to review.

### Packaging

Finch uses [poetry](https://python-poetry.org/) for packaging.

To install for development, clone the repository and run:
```bash
poetry install --extras test
```
to install the current project and dev dependencies.

### Publishing

The "Publish" GitHub Action is a manual workflow for publishing Python packages to PyPI using Poetry. It handles the version management based on the `pyproject.toml` file and automates tagging and creating GitHub releases.

#### Version Update

Before initiating the "Publish" action, update the package's version number in `pyproject.toml`. Follow semantic versioning guidelines for this update.

#### Triggering the Action

The action is triggered manually. Once the version in `pyproject.toml` is updated, manually start the "Publish" action from the GitHub repository's Actions tab.

#### Process and Outcomes

On successful execution, the action publishes the package to PyPI and tags the release in the GitHub repository. If the version number is not updated, the action fails to publish to PyPI, and no tagging or release is done. In case of failure, correct the version number and rerun the action.

#### Best Practices

- Ensure the version number in `pyproject.toml` is updated before triggering the action.
- Regularly check action logs for successful completion or to identify issues.

### Pre-commit hooks

Pull requests must pass some formatting, linting, and typing checks before we can merge them. These checks can be run automatically before you make commits, which is why they are sometimes called "pre-commit hooks". We use [pre-commit](https://pre-commit.com/) to run these checks.

To install pre-commit hooks to run before committing, run:
```bash
poetry run pre-commit install
```
If you prefer to instead run pre-commit hooks manually, run:
```bash
poetry run pre-commit run -a
```

### Testing

Finch uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run the
tests:

```bash
poetry run pytest
```
