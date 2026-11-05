"""Data ingestion and processing modules."""

from collisionguard.data.dataset import TaskRecord, sanitize_task_record, verify_temporal_validity
from collisionguard.data.ingestion import stream_commit_records, stream_linked_task_records
from collisionguard.data.merge_replay import MergeReplayResult, evaluate_merge_tree, find_merge_base

__all__ = [
    "stream_linked_task_records",
    "stream_commit_records",
    "MergeReplayResult",
    "evaluate_merge_tree",
    "find_merge_base",
    "TaskRecord",
    "verify_temporal_validity",
    "sanitize_task_record",
]


