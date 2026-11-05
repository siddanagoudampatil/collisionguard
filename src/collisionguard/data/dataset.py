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
