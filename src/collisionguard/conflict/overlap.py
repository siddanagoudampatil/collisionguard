from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

MANIFEST_FILENAMES: Set[str] = {
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pom.xml",
    "cargo.toml",
    "cargo.lock",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "go.mod",
    "go.sum",
    "build.gradle",
    "gemfile",
    "cmakelists.txt",
    "dockerfile",
    "makefile",
}

STAGE2_FEATURE_COLUMNS: List[str] = [
    "expected_overlap",
    "p_any_overlap",
    "manifest_overlap",
    "max_single_file_overlap",
    "jaccard_candidate_overlap",
    "issue_text_similarity",
    "repo_churn_14d",
    "agent_heterogeneity",
    "duration_overlap_hours",
]


@dataclass(frozen=True)
class PairwiseOverlapFeatures:
    """Pairwise features computed between two tasks for Stage 2 conflict risk estimation."""

    features: Dict[str, float]

    def to_array(self) -> np.ndarray:
        """Return features ordered according to STAGE2_FEATURE_COLUMNS."""
        return np.array([self.features[col] for col in STAGE2_FEATURE_COLUMNS], dtype=np.float32)


def is_manifest_file(file_path: str) -> bool:
    """Check if file is a build configuration, dependency, or manifest file."""
    name = Path(file_path).name.lower()
    return name in MANIFEST_FILENAMES


def sanitize_prob(p: Any) -> float:
    """Sanitize candidate probability, handling NaNs, Infinities, and boundary clipping."""
    try:
        val = float(p)
        if not np.isfinite(val):
            return 0.0
        return float(np.clip(val, 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def compute_pairwise_overlap_features(
    task_a_probs: Dict[str, float],
    task_b_probs: Dict[str, float],
    issue_text_similarity: float = 0.0,
    repo_churn_14d: float = 0.0,
    agent_heterogeneity: float = 0.0,
    duration_overlap_hours: float = 0.0,
) -> PairwiseOverlapFeatures:
    """Compute probabilistic footprint overlap and contextual conflict features between tasks A and B.

    Formulation:
    - Expected shared files: E[|F_A ∩ F_B|] = ∑_{f ∈ C_A ∩ C_B} P(f ∈ F_A) * P(f ∈ F_B)
    - Probability of any overlap: P(overlap ≥ 1) = 1 - ∏_{f ∈ C_A ∩ C_B} (1 - P(f ∈ F_A) * P(f ∈ F_B))
      Computed in log-space using log1p and expm1 for numerical stability against underflow.
    - Manifest overlap: E[shared manifests] = ∑_{f ∈ Manifests} P(f ∈ F_A) * P(f ∈ F_B)
    """
    if not task_a_probs or not task_b_probs:
        return PairwiseOverlapFeatures(
            features={
                "expected_overlap": 0.0,
                "p_any_overlap": 0.0,
                "manifest_overlap": 0.0,
                "max_single_file_overlap": 0.0,
                "jaccard_candidate_overlap": 0.0,
                "issue_text_similarity": float(np.clip(issue_text_similarity, -1.0, 1.0)),
                "repo_churn_14d": float(max(0.0, repo_churn_14d)),
                "agent_heterogeneity": float(1.0 if agent_heterogeneity else 0.0),
                "duration_overlap_hours": float(max(0.0, duration_overlap_hours)),
            }
        )

    candidates_a = set(task_a_probs.keys())
    candidates_b = set(task_b_probs.keys())
    common_files = candidates_a & candidates_b
    union_files = candidates_a | candidates_b

    jaccard = float(len(common_files)) / float(len(union_files)) if union_files else 0.0

    # Edge case: zero common candidates
    if not common_files:
        return PairwiseOverlapFeatures(
            features={
                "expected_overlap": 0.0,
                "p_any_overlap": 0.0,
                "manifest_overlap": 0.0,
                "max_single_file_overlap": 0.0,
                "jaccard_candidate_overlap": 0.0,
                "issue_text_similarity": float(np.clip(issue_text_similarity, -1.0, 1.0)),
                "repo_churn_14d": float(max(0.0, repo_churn_14d)),
                "agent_heterogeneity": float(1.0 if agent_heterogeneity else 0.0),
                "duration_overlap_hours": float(max(0.0, duration_overlap_hours)),
            }
        )

    expected_overlap = 0.0
    manifest_overlap = 0.0
    max_single_file = 0.0
    log_complement_sum = 0.0

    for f in common_files:
        p_a = sanitize_prob(task_a_probs[f])
        p_b = sanitize_prob(task_b_probs[f])
        joint_prob = p_a * p_b

        if joint_prob <= 0.0:
            continue

        expected_overlap += joint_prob

        # Numerically stable log1p: log(1 - joint_prob)
        # Avoid log(0) if joint_prob == 1.0
        clamped_joint = min(joint_prob, 1.0 - 1e-12)
        log_complement_sum += float(np.log1p(-clamped_joint))

        if joint_prob > max_single_file:
            max_single_file = joint_prob

        if is_manifest_file(f):
            manifest_overlap += joint_prob

    # 1 - exp(sum(log(1 - p_i))) = -expm1(sum_log)
    p_any_overlap = float(-np.expm1(log_complement_sum))
    p_any_overlap = float(np.clip(p_any_overlap, 0.0, 1.0))

    feature_dict = {
        "expected_overlap": float(expected_overlap),
        "p_any_overlap": p_any_overlap,
        "manifest_overlap": float(manifest_overlap),
        "max_single_file_overlap": float(max_single_file),
        "jaccard_candidate_overlap": float(np.clip(jaccard, 0.0, 1.0)),
        "issue_text_similarity": float(np.clip(issue_text_similarity, -1.0, 1.0)),
        "repo_churn_14d": float(max(0.0, repo_churn_14d)),
        "agent_heterogeneity": float(1.0 if agent_heterogeneity else 0.0),
        "duration_overlap_hours": float(max(0.0, duration_overlap_hours)),
    }

    return PairwiseOverlapFeatures(features=feature_dict)
