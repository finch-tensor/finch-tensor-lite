# AI modified: 2025-01-01T00:00:00Z parent=4f5a2e5021678965ce8d830bb9edecac1dd3fea9
# AI modified: 2025-01-01T00:01:00Z parent=4f5a2e5021678965ce8d830bb9edecac1dd3fea9
# AI modified: 2026-04-02T22:59:00Z parent=197d5a907823d2a53fcd3b68b674f3f4d4f50b5d
# AI modified: 2026-04-03T15:30:00Z parent=36276c257318d74488f81fa8107d2f2d0a8b804c
import os
import re
import subprocess
import sys
from pathlib import Path

import finchlite


def _run_codegen_session(
    repo_root: Path,
    data_path: Path,
    python_code: str,
):
    env = os.environ.copy()
    env["FINCHLITE_DATA_PATH"] = str(data_path)
    return subprocess.run(
        [sys.executable, "-c", python_code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def _restore_file_state(path: Path, *, contents: str, atime_ns: int, mtime_ns: int):
    path.write_text(contents)
    os.utime(path, ns=(atime_ns, mtime_ns))


def test_c_codegen_cache_invalidation_end_to_end_across_sessions(tmp_path):
    # finchlite.__path__[0] points to <repo>/src/finchlite, so parent.parent is repo root.
    repo_root = Path(finchlite.__path__[0]).parent.parent
    data_path = tmp_path / "finch_data"
    c_codegen_file = repo_root / "src" / "finchlite" / "codegen" / "c_codegen.py"
    stat_before = c_codegen_file.stat()
    original_contents = c_codegen_file.read_text()
    base_script = """
import finchlite.codegen.c_codegen as ccg

code = '''
int unique_value() {
    return 7;
}
'''
ccg.load_shared_lib.cache_clear()
lib = ccg.load_shared_lib(code)
print("RESULT", int(lib.unique_value()))
"""

    first = _run_codegen_session(repo_root, data_path, base_script)
    first_result = int(first.stdout.strip().split("RESULT ", 1)[1])
    assert first_result == 7

    try:
        modified_contents = re.sub(
            r"c_file_path\.write_text\(c_code\)",
            'c_file_path.write_text(c_code.replace("return 7;", "return 9;"))',
            original_contents,
            count=1,
        )
        c_codegen_file.write_text(modified_contents)

        second = _run_codegen_session(repo_root, data_path, base_script)
        second_result = int(second.stdout.strip().split("RESULT ", 1)[1])
        assert second_result == 9
    finally:
        _restore_file_state(
            c_codegen_file,
            contents=original_contents,
            atime_ns=stat_before.st_atime_ns,
            mtime_ns=stat_before.st_mtime_ns,
        )
