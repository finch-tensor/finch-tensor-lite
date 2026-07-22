import os
import shlex
import subprocess
import sys
from pathlib import Path


def test_array_api():
    array_api_tests_dir = Path(
        os.environ.get(
            "ARRAY_API_TESTS_DIR",
            Path(__file__).parent.parent / "array-api-tests",
        ),
    ).resolve()
    array_api_tests_rev = os.environ.get(
        "ARRAY_API_TESTS_REV", "c48410f96fc58e02eea844e6b7f6cc01680f77ce"
    )
    array_api_tests_skips = Path(
        os.environ.get(
            "ARRAY_API_TESTS_SKIPS",
            Path(__file__).parent.parent / "array-api-skips.txt",
        ),
    ).resolve()
    array_api_tests_args = shlex.split(os.environ.get("ARRAY_API_TESTS_ARGS", "-vv -s"))

    print(f"[array-api] using dir: {array_api_tests_dir}", flush=True)
    print(f"[array-api] target rev: {array_api_tests_rev}", flush=True)

    if not (array_api_tests_dir / ".git").exists():
        if array_api_tests_dir.exists() and any(array_api_tests_dir.iterdir()):
            raise RuntimeError(
                f"{array_api_tests_dir} exists but is not a git checkout. "
                "Set ARRAY_API_TESTS_DIR to a valid checkout or an empty path."
            )
        print("[array-api] cloning test repo...", flush=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--recurse-submodules",
                "https://github.com/data-apis/array-api-tests.git",
                str(array_api_tests_dir),
            ],
            check=True,
        )

    array_api_tests_git = [
        "git",
        "-C",
        str(array_api_tests_dir),
    ]

    print("[array-api] cleaning repo...", flush=True)
    subprocess.run(
        [
            *array_api_tests_git,
            "clean",
            "-xddf",
        ],
        check=True,
    )

    print("[array-api] fetching latest refs...", flush=True)
    subprocess.run(
        [
            *array_api_tests_git,
            "fetch",
        ],
        check=False,
    )

    print("[array-api] checking out target revision...", flush=True)
    subprocess.run(
        [
            *array_api_tests_git,
            "reset",
            "--hard",
            array_api_tests_rev,
        ],
        check=True,
    )

    print("[array-api] initializing submodules...", flush=True)
    subprocess.run(
        [
            *array_api_tests_git,
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        check=True,
    )

    # Run the tests using pytest
    print("[array-api] running external array-api-tests...", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(array_api_tests_dir / "pytest.ini"),
            *array_api_tests_args,
            str(array_api_tests_dir / "array_api_tests"),
            "--max-examples=2",
            "--derandomize",
            "--disable-deadline",
            "--disable-warnings",
            "--skips-file",
            str(array_api_tests_skips),
        ],
        env={
            **os.environ,
            "ARRAY_API_TESTS_MODULE": "finch",
            "PYTEST_ADDOPTS": "",
            "PYTHONUNBUFFERED": "1",
        },
        check=False,
        text=True,
    )
    print("[array-api] array-api-tests completed!", flush=True)
    assert result.returncode == 0, "Array API tests failed"
