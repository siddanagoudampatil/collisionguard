import subprocess
from pathlib import Path
import pytest

from collisionguard.data.merge_replay import evaluate_merge_tree, find_merge_base


def _run_git(cwd: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(cwd), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return res.stdout.strip()


@pytest.fixture
def synthetic_git_repo(tmp_path: Path):
    """Build a temporary git repository with divergent clean and conflicting branches."""
    repo_dir = tmp_path / "synthetic_repo"
    repo_dir.mkdir()

    _run_git(repo_dir, "init")
    _run_git(repo_dir, "config", "user.name", "TestUser")
    _run_git(repo_dir, "config", "user.email", "test@test.local")
    _run_git(repo_dir, "branch", "-M", "main")

    # Initial base commit
    clean_file = repo_dir / "clean.py"
    conflict_file = repo_dir / "conflict.py"
    clean_file.write_text("def base():\n    return 0\n", encoding="utf-8")
    conflict_file.write_text("BASE = 1\n", encoding="utf-8")

    _run_git(repo_dir, "add", ".")
    _run_git(repo_dir, "commit", "-m", "Base commit")
    base_sha = _run_git(repo_dir, "rev-parse", "HEAD")

    # Branch A: modifies clean.py and conflict.py
    _run_git(repo_dir, "checkout", "-b", "branch-a")
    clean_file.write_text("def base():\n    # branch A comment\n    return 0\n", encoding="utf-8")
    conflict_file.write_text("BASE = 2  # Branch A change\n", encoding="utf-8")
    _run_git(repo_dir, "commit", "-am", "Branch A update")
    sha_a = _run_git(repo_dir, "rev-parse", "HEAD")

    # Branch B: modifies only conflict.py on same line
    _run_git(repo_dir, "checkout", "main")
    _run_git(repo_dir, "checkout", "-b", "branch-b")
    conflict_file.write_text("BASE = 3  # Branch B change\n", encoding="utf-8")
    _run_git(repo_dir, "commit", "-am", "Branch B update")
    sha_b = _run_git(repo_dir, "rev-parse", "HEAD")

    # Branch Clean: modifies only clean.py on another file
    _run_git(repo_dir, "checkout", "main")
    _run_git(repo_dir, "checkout", "-b", "branch-clean")
    new_file = repo_dir / "independent.py"
    new_file.write_text("NEW = True\n", encoding="utf-8")
    _run_git(repo_dir, "add", ".")
    _run_git(repo_dir, "commit", "-m", "Clean independent update")
    sha_clean = _run_git(repo_dir, "rev-parse", "HEAD")

    return {
        "repo_dir": repo_dir,
        "base_sha": base_sha,
        "sha_a": sha_a,
        "sha_b": sha_b,
        "sha_clean": sha_clean,
    }


def test_find_merge_base(synthetic_git_repo):
    repo = synthetic_git_repo["repo_dir"]
    base = find_merge_base(repo, synthetic_git_repo["sha_a"], synthetic_git_repo["sha_b"])
    assert base == synthetic_git_repo["base_sha"]


def test_clean_merge_evaluation(synthetic_git_repo):
    repo = synthetic_git_repo["repo_dir"]
    result = evaluate_merge_tree(repo, synthetic_git_repo["sha_a"], synthetic_git_repo["sha_clean"])
    assert result.is_valid is True
    assert result.label == 0
    assert result.conflict_files == []
    assert result.base_commit == synthetic_git_repo["base_sha"]


def test_conflict_merge_evaluation(synthetic_git_repo):
    repo = synthetic_git_repo["repo_dir"]
    result = evaluate_merge_tree(repo, synthetic_git_repo["sha_a"], synthetic_git_repo["sha_b"])
    assert result.is_valid is True
    assert result.label == 1
    assert "conflict.py" in result.conflict_files
    assert result.base_commit == synthetic_git_repo["base_sha"]


def test_invalid_revision_handling(synthetic_git_repo):
    repo = synthetic_git_repo["repo_dir"]
    bad_sha = "0000000000000000000000000000000000000000"
    result = evaluate_merge_tree(repo, synthetic_git_repo["sha_a"], bad_sha)
    assert result.is_valid is False
    assert result.label == -1
