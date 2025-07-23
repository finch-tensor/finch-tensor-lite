## Regression Testing
This project uses pytest-regressions to support testing compiler outputs.
See [docs](https://pytest-regressions.readthedocs.io/en/latest/overview.html) for getting started
or look at the examples in test files.
### Key Points
Beyond the basic pytest-regressions usage, this project has some customizations for program regression testing:
- **Program Regression**: The `program_regression` fixture can be used to test generated programs that are in the form of trees as well as programs in plain string. It builds on the `file_regression` fixture to add support for non-deterministic outputs like memory addresses of functions in program trees. You can specify a custom formatter function to convert your prgram to a string representation. By default, it uses `pprint.pformat`.
- **Cleanup**: A hook is provided that automatically cleans up any *obtained* files of a test in case it passes. If the test fails, files are preserved for debugging.
- **Preserving Obtained Files**: The `preserve_obtained` fixture can be used to preserve the obtained files even if the test passes.
- Avoid using a `basename` in `..._regression.check()` calls. It will throw off the automatic cleanup mechanism and you'll have to manually clean up the files after the test run. Not using the paramter will cause pytest-regressions to place the files in a directory named after the test function, so everything is organized.
- use `--force-regen` flag to regenerate the regression files. This is useful when you want to update the expected outputs after making changes to the compiler logic.
