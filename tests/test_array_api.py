import os
import subprocess
import sys


def test_array_api():
    ARRAY_API_TESTS_DIR = os.environ.get(
        "ARRAY_API_TESTS_DIR",
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../array-api-tests"),
        ),
    )
    ARRAY_API_TESTS_REV = os.environ.get(
        "ARRAY_API_TESTS_REV", "c48410f96fc58e02eea844e6b7f6cc01680f77ce"
    )
    ARRAY_API_TESTS_SKIPS = os.environ.get(
        "ARRAY_API_TESTS_SKIPS",
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../array-api-skips.txt"),
        ),
    )
    ARRAY_API_TESTS_ARGS = os.environ.get("ARRAY_API_TESTS_ARGS", "-vv -s")

    print(f"[array-api] using dir: {ARRAY_API_TESTS_DIR}", flush=True)
    print(f"[array-api] target rev: {ARRAY_API_TESTS_REV}", flush=True)

    if not os.path.isdir(ARRAY_API_TESTS_DIR):
        print("[array-api] cloning test repo...", flush=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--recursive",
                "https://github.com/data-apis/array-api-tests.git",
                ARRAY_API_TESTS_DIR,
            ],
            check=True,
        )

    print("[array-api] cleaning repo...", flush=True)
    subprocess.run(
        [
            "git",
            "--git-dir",
            f"{ARRAY_API_TESTS_DIR}/.git",
            "--work-tree",
            ARRAY_API_TESTS_DIR,
            "clean",
            "-xddf",
        ],
        check=True,
    )

    print("[array-api] fetching latest refs...", flush=True)
    subprocess.run(
        [
            "git",
            "--git-dir",
            f"{ARRAY_API_TESTS_DIR}/.git",
            "--work-tree",
            ARRAY_API_TESTS_DIR,
            "fetch",
        ],
        check=True,
    )

    print("[array-api] checking out target revision...", flush=True)
    subprocess.run(
        [
            "git",
            "--git-dir",
            f"{ARRAY_API_TESTS_DIR}/.git",
            "--work-tree",
            ARRAY_API_TESTS_DIR,
            "reset",
            "--hard",
            ARRAY_API_TESTS_REV,
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
            *ARRAY_API_TESTS_ARGS.split(),
            f"{ARRAY_API_TESTS_DIR}/array_api_tests/",
            "--max-examples=2",
            "--derandomize",
            "--disable-deadline",
            "--disable-warnings",
            "--skips-file",
            ARRAY_API_TESTS_SKIPS,
        ],
        env={
            **os.environ,
            "ARRAY_API_TESTS_MODULE": "finchlite",
            "PYTHONUNBUFFERED": "1",
        },
        check=False,
        text=True,
    )
    print("[array-api] array-api-tests completed!", flush=True)
    assert result.returncode == 0, "Array API tests failed"
