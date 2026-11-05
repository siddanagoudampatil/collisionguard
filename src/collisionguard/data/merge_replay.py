from dataclasses import dataclass, field
import logging
from pathlib import Path
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeReplayResult:
    """Represents the outcome of a headless 3-way Git merge replay."""

    label: int
    conflict_files: List[str] = field(default_factory=list)
    base_commit: Optional[str] = None
    is_valid: bool = True
    error_message: Optional[str] = None


def find_merge_base(repo_dir: str | Path, commit_a: str, commit_b: str) -> Optional[str]:
    """Identify lowest common ancestor commit for two Git revisions."""
    cmd = ["git", "-C", str(repo_dir), "merge-base", commit_a, commit_b]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            base = proc.stdout.strip()
            return base if base else None
        return None
    except Exception as exc:
        logger.error("Failed to resolve merge-base between %s and %s: %s", commit_a, commit_b, exc)
        return None


def evaluate_merge_tree(repo_dir: str | Path, commit_a: str, commit_b: str) -> MergeReplayResult:
    """Evaluate 3-way merge conflict headlessly using git merge-tree.

    Contract:
    - Exit code 0: Clean merge (label = 0), no conflicting files.
    - Exit code 1: Merge conflict detected (label = 1), stdout contains conflict file list.
    - Exit code > 1: Repository corruption, missing refs, or syntax error (is_valid = False).
    """
    base = find_merge_base(repo_dir, commit_a, commit_b)
    if not base:
        return MergeReplayResult(
            label=-1,
            is_valid=False,
            error_message=f"Missing common ancestor commit between {commit_a} and {commit_b}",
        )

    # Modern Git (2.38+) 3-way syntax
    modern_cmd = [
        "git",
        "-C",
        str(repo_dir),
        "merge-tree",
        "--write-tree",
        "--no-messages",
        "--name-only",
        commit_a,
        commit_b,
    ]

    try:
        proc = subprocess.run(
            modern_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        # If modern syntax is supported
        if proc.returncode in (0, 1) and "usage:" not in proc.stderr:
            if proc.returncode == 0:
                return MergeReplayResult(
                    label=0,
                    conflict_files=[],
                    base_commit=base,
                    is_valid=True,
                )
            # Exit code 1 indicates merge conflicts were encountered.
            raw_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            conflicting_files = [line for line in raw_lines if not (len(line) == 40 and all(c in "0123456789abcdef" for c in line))]
            if not conflicting_files and raw_lines:
                conflicting_files = raw_lines[1:] if len(raw_lines) > 1 else raw_lines
            return MergeReplayResult(
                label=1,
                conflict_files=sorted(list(set(conflicting_files))),
                base_commit=base,
                is_valid=True,
            )

        # Legacy Git (<= 2.37) syntax fallback: git merge-tree <base-tree> <branch1> <branch2>
        legacy_cmd = ["git", "-C", str(repo_dir), "merge-tree", base, commit_a, commit_b]
        legacy_proc = subprocess.run(
            legacy_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if legacy_proc.returncode != 0:
            return MergeReplayResult(
                label=-1,
                is_valid=False,
                base_commit=base,
                error_message=legacy_proc.stderr.strip() or f"git merge-tree failed with code {legacy_proc.returncode}",
            )

        stdout = legacy_proc.stdout
        # Conflict presence is identified by conflict markers or changed in both status
        has_conflict = "+<<<<<<<" in stdout or "changed in both" in stdout
        if not has_conflict:
            return MergeReplayResult(
                label=0,
                conflict_files=[],
                base_commit=base,
                is_valid=True,
            )

        # Extract conflicting filenames from "changed in both" blocks
        conflict_files = []
        lines = stdout.splitlines()
        for idx, line in enumerate(lines):
            if "changed in both" in line:
                for sub_line in lines[idx + 1 : idx + 5]:
                    parts = sub_line.strip().split()
                    if len(parts) >= 3 and parts[0] in ("our", "their", "base"):
                        conflict_files.append(parts[-1])
                        break

        return MergeReplayResult(
            label=1,
            conflict_files=sorted(list(set(conflict_files))),
            base_commit=base,
            is_valid=True,
        )

    except Exception as exc:
        return MergeReplayResult(
            label=-1,
            is_valid=False,
            base_commit=base,
            error_message=str(exc),
        )

