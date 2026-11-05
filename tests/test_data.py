from datetime import datetime, timezone
import pandas as pd
import pytest

from collisionguard.data.dataset import (
    TaskRecord,
    build_concurrent_pairs,
    clean_posix_path,
    sanitize_task_record,
    verify_temporal_validity,
)


def test_clean_posix_path():
    assert clean_posix_path("src\\utils\\helpers.py") == "src/utils/helpers.py"
    assert clean_posix_path("/leading/slash.py") == "leading/slash.py"
    assert clean_posix_path("M\tsrc/modified.py") == "src/modified.py"
    assert clean_posix_path("CONFLICT (content): Merge conflict in lib/core.py") == "lib/core.py"
    assert clean_posix_path("Auto-merging src/index.ts") == "src/index.ts"


def test_sanitize_task_record_path_normalization():
    raw = {
        "pr_id": "PR_200",
        "repo_url": "https://github.com/org/repo",
        "head_sha": "abc1234",
        "issue_id": "ISSUE_200",
        "issue_title": "Update paths",
        "issue_body": "Normalize Windows paths",
        "issue_created_at": "2026-01-01T10:00:00Z",
        "pr_created_at": "2026-01-01T12:00:00Z",
        "modified_files": ["src\\windows\\file.py", "M\tsrc/diff.py"],
    }
    rec = sanitize_task_record(raw)
    assert rec is not None
    assert rec.modified_files == ["src/windows/file.py", "src/diff.py"]



def test_verify_temporal_validity():
    t_issue = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    t_pr_valid = datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc)
    t_pr_equal = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    t_pr_invalid = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)

    assert verify_temporal_validity(t_issue, t_pr_valid) is True
    assert verify_temporal_validity(t_issue, t_pr_equal) is True
    assert verify_temporal_validity(t_issue, t_pr_invalid) is False


def test_sanitize_task_record_filters_temporal_leakage():
    valid_raw = {
        "pr_id": "PR_101",
        "repo_url": "https://github.com/org/repo",
        "head_sha": "abc1234",
        "issue_id": "ISSUE_1",
        "issue_title": "Fix memory leak in parser",
        "issue_body": "Parser fails to close buffer on EOF.",
        "issue_created_at": "2026-01-01T10:00:00Z",
        "pr_created_at": "2026-01-01T12:00:00Z",
        "pr_closed_at": "2026-01-01T15:00:00Z",
        "modified_files": ["parser.py"],
    }
    record = sanitize_task_record(valid_raw)
    assert record is not None
    assert record.task_id == "PR_101"
    assert record.issue_title == "Fix memory leak in parser"

    invalid_raw = dict(valid_raw)
    invalid_raw["issue_created_at"] = "2026-01-01T14:00:00Z"
    invalid_raw["pr_created_at"] = "2026-01-01T11:00:00Z"
    assert sanitize_task_record(invalid_raw) is None


def test_sanitize_task_record_empty_body_and_title():
    empty_raw = {
        "pr_id": "PR_102",
        "repo_url": "https://github.com/org/repo",
        "head_sha": "def5678",
        "issue_id": "ISSUE_2",
        "issue_title": "   ",
        "issue_body": "",
        "issue_created_at": "2026-01-01T10:00:00Z",
        "pr_created_at": "2026-01-01T12:00:00Z",
    }
    assert sanitize_task_record(empty_raw) is None


def test_build_concurrent_pairs_overlap_and_capping():
    def make_task(task_id: str, start_h: int, end_h: int) -> TaskRecord:
        return TaskRecord(
            task_id=task_id,
            repo_url="https://github.com/org/repoA",
            head_sha=f"sha_{task_id}",
            issue_id=f"iss_{task_id}",
            issue_title=f"Task {task_id}",
            issue_body="Description",
            issue_created_at=pd.to_datetime("2026-01-01T00:00:00Z"),
            pr_created_at=pd.to_datetime(f"2026-01-01T{start_h:02d}:00:00Z"),
            pr_closed_at=pd.to_datetime(f"2026-01-01T{end_h:02d}:00:00Z"),
            modified_files=["foo.py"],
        )

    t1 = make_task("1", 2, 6)
    t2 = make_task("2", 4, 8)  # overlaps with t1 from 4 to 6
    t3 = make_task("3", 7, 10)  # overlaps with t2 from 7 to 8, disjoint with t1

    pairs = build_concurrent_pairs([t1, t2, t3], max_pairs_per_repo=30)
    assert len(pairs) == 2

    pair_ids = {p.pair_id for p in pairs}
    assert "1_2" in pair_ids
    assert "2_3" in pair_ids
    assert "1_3" not in pair_ids

    # Check capping behavior
    many_tasks = [make_task(str(i), 1, 10) for i in range(15)]
    capped_pairs = build_concurrent_pairs(many_tasks, max_pairs_per_repo=5)
    assert len(capped_pairs) == 5
