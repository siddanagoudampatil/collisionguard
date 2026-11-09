from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from collisionguard.data.dataset import ConcurrentTaskPair, TaskRecord

logger = logging.getLogger(__name__)


class ConflictCategory(str, Enum):
    """Structural categories of Git merge conflicts."""

    CONTENT = "content"
    MODIFY_DELETE = "modify_delete"
    ADD_ADD = "add_add"
    CLEAN = "clean"


def parse_merge_tree_conflicts(stdout: str) -> List[Tuple[ConflictCategory, str]]:
    """Parse git merge-tree stdout into structural conflict categories and file paths.

    Handles both Git 2.38+ messages format and legacy 3-way text outputs:
    - CONFLICT (content): Merge conflict in <file>
    - CONFLICT (modify/delete): <file> deleted in ... and modified in ...
    - CONFLICT (add/add): Merge conflict in <file>
    - Legacy 'changed in both', 'added in both', 'deleted in our/their'
    """
    if not stdout:
        return []

    results: List[Tuple[ConflictCategory, str]] = []
    lines = stdout.splitlines()

    for idx, line in enumerate(lines):
        line_clean = line.strip()

        # 1. Modify/Delete conflicts
        if "CONFLICT (modify/delete):" in line_clean:
            match = re.search(r"CONFLICT \(modify/delete\):\s*([^\s]+)", line_clean)
            file_path = match.group(1).strip() if match else ""
            if file_path:
                results.append((ConflictCategory.MODIFY_DELETE, file_path))
            continue
        if "deleted in" in line_clean and ("modified in" in line_clean or "changed" in line_clean):
            parts = line_clean.split()
            path = parts[-1] if parts else ""
            if path:
                results.append((ConflictCategory.MODIFY_DELETE, path))
            continue

        # 2. Add/Add conflicts
        if "CONFLICT (add/add):" in line_clean:
            match = re.search(r"Merge conflict in\s+([^\s]+)", line_clean) or re.search(
                r"CONFLICT \(add/add\):\s*([^\s]+)", line_clean
            )
            file_path = match.group(1).strip() if match else ""
            if file_path:
                results.append((ConflictCategory.ADD_ADD, file_path))
            continue
        if "added in both" in line_clean:
            # Look ahead for file name
            for next_line in lines[idx + 1 : idx + 4]:
                parts = next_line.strip().split()
                if len(parts) >= 3 and parts[0] in ("our", "their", "base"):
                    results.append((ConflictCategory.ADD_ADD, parts[-1]))
                    break
            continue

        # 3. Content conflicts
        if "CONFLICT (content):" in line_clean:
            match = re.search(r"Merge conflict in\s+([^\s]+)", line_clean)
            file_path = match.group(1).strip() if match else ""
            if file_path:
                results.append((ConflictCategory.CONTENT, file_path))
            continue
        if "changed in both" in line_clean:
            for next_line in lines[idx + 1 : idx + 4]:
                parts = next_line.strip().split()
                if len(parts) >= 3 and parts[0] in ("our", "their", "base"):
                    results.append((ConflictCategory.CONTENT, parts[-1]))
                    break
            continue

    return results


def classify_pair_conflict(
    pair: Union[ConcurrentTaskPair, Any],
    merge_stdout: Optional[str] = None,
) -> ConflictCategory:
    """Classify a concurrent task pair into a primary structural conflict category."""
    # If merge-tree stdout is provided, extract directly
    if merge_stdout:
        parsed = parse_merge_tree_conflicts(merge_stdout)
        if parsed:
            categories = [cat for cat, _ in parsed]
            if ConflictCategory.MODIFY_DELETE in categories:
                return ConflictCategory.MODIFY_DELETE
            if ConflictCategory.ADD_ADD in categories:
                return ConflictCategory.ADD_ADD
            if ConflictCategory.CONTENT in categories:
                return ConflictCategory.CONTENT

    # Check pair label
    is_conflict = (
        pair.label == 1
        if hasattr(pair, "label") and pair.label is not None
        else getattr(pair, "is_conflict", False)
    )
    if not is_conflict:
        return ConflictCategory.CLEAN

    # Fallback heuristic: file modifications
    task_a = getattr(pair, "task_a", None)
    task_b = getattr(pair, "task_b", None)
    if task_a and task_b:
        files_a = set(getattr(task_a, "modified_files", []))
        files_b = set(getattr(task_b, "modified_files", []))
        if files_a & files_b:
            return ConflictCategory.CONTENT

    return ConflictCategory.CONTENT


def compute_taxonomy_metrics(
    y_true_categories: List[Union[ConflictCategory, str]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1, and PR-AUC breakdown across structural conflict types.

    Args:
        y_true_categories: List of ground-truth ConflictCategory strings or enum members.
        y_prob: Model predicted probabilities of conflict P(Conflict).
        threshold: Decision threshold for computing precision and recall.

    Returns:
        Dictionary mapping conflict category names to performance metrics.
    """
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred = (y_p >= threshold).astype(int)

    cat_strs = [
        c.value if isinstance(c, ConflictCategory) else str(c).lower()
        for c in y_true_categories
    ]

    target_categories = [
        ConflictCategory.CONTENT.value,
        ConflictCategory.MODIFY_DELETE.value,
        ConflictCategory.ADD_ADD.value,
    ]

    breakdown: Dict[str, Dict[str, float]] = {}

    for cat in target_categories:
        binary_true = np.array([1 if c == cat else 0 for c in cat_strs], dtype=int)
        support = int(np.sum(binary_true))

        if support == 0:
            breakdown[cat] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "pr_auc": 0.0,
                "support": 0,
            }
            continue

        # Precision, recall, and F1 at threshold
        p, r, f1, _ = precision_recall_fscore_support(
            binary_true, y_pred, average="binary", zero_division=0.0
        )

        # PR-AUC for this category
        if len(np.unique(binary_true)) > 1:
            pr_auc = float(average_precision_score(binary_true, y_p))
        else:
            pr_auc = float(p)

        breakdown[cat] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "pr_auc": round(float(pr_auc), 4),
            "support": support,
        }

    return breakdown
