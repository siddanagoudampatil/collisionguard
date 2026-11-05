from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskRecord:
    """Represents an atomic issue-linked task execution instance."""

    task_id: str
    repo_url: str
    head_sha: str
    issue_id: str
    issue_title: str
    issue_body: str
    issue_created_at: pd.Timestamp
    pr_created_at: pd.Timestamp
    pr_closed_at: pd.Timestamp
    modified_files: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConcurrentTaskPair:
    """Represents a co-active pair of tasks executing concurrently in a repository."""

    pair_id: str
    repo_url: str
    task_a: TaskRecord
    task_b: TaskRecord
    overlap_duration_hours: float
    base_commit: Optional[str] = None
    label: Optional[int] = None
    conflict_files: List[str] = field(default_factory=list)


def verify_temporal_validity(
    issue_created_at: str | datetime | pd.Timestamp,
    pr_created_at: str | datetime | pd.Timestamp,
) -> bool:
    """Enforce strict temporal validity: T_issue <= T_pr_open.

    Tasks whose issue was created after the PR was opened represent post-hoc
    retroactive tracking and violate pre-execution prediction assumptions.
    """
    try:
        ts_issue = pd.to_datetime(issue_created_at, utc=True)
        ts_pr = pd.to_datetime(pr_created_at, utc=True)
        return bool(ts_issue <= ts_pr)
    except Exception as exc:
        logger.warning("Timestamp parsing failed during temporal check: %s", exc)
        return False


def sanitize_task_record(raw: Dict[str, Any]) -> Optional[TaskRecord]:
    """Parse raw dictionary into TaskRecord, enforcing temporal leakage filters."""
    issue_time = raw.get("issue_created_at")
    pr_time = raw.get("pr_created_at")

    if not issue_time or not pr_time:
        return None

    if not verify_temporal_validity(issue_time, pr_time):
        logger.debug(
            "Rejecting task %s due to temporal leakage: issue_created (%s) > pr_created (%s)",
            raw.get("pr_id"),
            issue_time,
            pr_time,
        )
        return None

    title = str(raw.get("issue_title") or "").strip()
    body = str(raw.get("issue_body") or "").strip()

    # Reject records where both title and body are empty
    if not title and not body:
        logger.debug("Rejecting task %s with empty issue title and body", raw.get("pr_id"))
        return None

    return TaskRecord(
        task_id=str(raw.get("pr_id") or raw.get("task_id")),
        repo_url=str(raw.get("repo_url") or "").strip(),
        head_sha=str(raw.get("head_sha") or "").strip(),
        issue_id=str(raw.get("issue_id") or "").strip(),
        issue_title=title,
        issue_body=body,
        issue_created_at=pd.to_datetime(issue_time, utc=True),
        pr_created_at=pd.to_datetime(pr_time, utc=True),
        pr_closed_at=pd.to_datetime(raw.get("pr_closed_at") or pr_time, utc=True),
        modified_files=list(raw.get("modified_files") or []),
    )


def build_concurrent_pairs(
    tasks: List[TaskRecord],
    max_pairs_per_repo: int = 30,
) -> List[ConcurrentTaskPair]:
    """Identify co-active task pairs within repositories.

    Two tasks P_A and P_B are co-active if their execution intervals overlap:
        [T_open,A, T_close,A] ∩ [T_open,B, T_close,B] ≠ ∅
    Capped per repository to prevent single-project dominance.
    """
    by_repo: Dict[str, List[TaskRecord]] = {}
    for t in tasks:
        by_repo.setdefault(t.repo_url, []).append(t)

    pairs: List[ConcurrentTaskPair] = []

    for repo_url, repo_tasks in by_repo.items():
        repo_pairs = []
        n = len(repo_tasks)
        for i in range(n):
            for j in range(i + 1, n):
                t_a = repo_tasks[i]
                t_b = repo_tasks[j]

                # Ensure canonical chronological ordering
                if t_a.pr_created_at > t_b.pr_created_at:
                    t_a, t_b = t_b, t_a

                # Calculate overlap interval
                start_overlap = max(t_a.pr_created_at, t_b.pr_created_at)
                end_overlap = min(t_a.pr_closed_at, t_b.pr_closed_at)

                if start_overlap <= end_overlap:
                    overlap_hours = (end_overlap - start_overlap).total_seconds() / 3600.0
                    pair = ConcurrentTaskPair(
                        pair_id=f"{t_a.task_id}_{t_b.task_id}",
                        repo_url=repo_url,
                        task_a=t_a,
                        task_b=t_b,
                        overlap_duration_hours=max(0.0, overlap_hours),
                    )
                    repo_pairs.append(pair)

                    if len(repo_pairs) >= max_pairs_per_repo:
                        break
            if len(repo_pairs) >= max_pairs_per_repo:
                break
        pairs.extend(repo_pairs)

    return pairs

