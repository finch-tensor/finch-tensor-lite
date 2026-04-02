# AI modified: 2025-01-01T00:00:00Z parent=4f5a2e5021678965ce8d830bb9edecac1dd3fea9
# AI modified: 2025-01-01T00:01:00Z parent=4f5a2e5021678965ce8d830bb9edecac1dd3fea9
import os
import subprocess
import sys
import time
from pathlib import Path

ONE_SECOND_NS = 1_000_000_000


def _read_cached_token(repo_root: Path, data_path: Path) -> str:
    script = """
import uuid
from pathlib import Path
from finchlite.util.cache import file_cache

@file_cache(ext="txt", domain="e2e_cache")
def write_cached(path):
    Path(path).write_text(str(uuid.uuid4()))

cache_file = write_cached()
print(Path(cache_file).read_text())
"""
    env = os.environ.copy()
    env["FINCHLITE_DATA_PATH"] = str(data_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout.strip()


def test_cache_invalidation_end_to_end_across_sessions(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    data_path = tmp_path / "finch_data"
    modified_file = repo_root / "src" / "finchlite" / "util" / "print.py"
    stat_before = modified_file.stat()

    try:
        token_1 = _read_cached_token(repo_root, data_path)
        token_2 = _read_cached_token(repo_root, data_path)
        assert token_2 == token_1

        new_mtime_ns = max(stat_before.st_mtime_ns + ONE_SECOND_NS, time.time_ns())
        os.utime(modified_file, ns=(stat_before.st_atime_ns, new_mtime_ns))

        token_3 = _read_cached_token(repo_root, data_path)
        assert token_3 != token_2
    finally:
        os.utime(
            modified_file, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns)
        )
