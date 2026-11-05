from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

MANIFEST_FILENAMES: Set[str] = {
    "package.json",
    "package-lock.json",
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
    - Manifest overlap: E[shared manifests] = ∑_{f ∈ Manifests} P(f ∈ F_A) * P(f ∈ F_B)
    """
    candidates_a = set(task_a_probs.keys())
    candidates_b = set(task_b_probs.keys())
    common_files = candidates_a & candidates_b
    union_files = candidates_a | candidates_b

    jaccard = float(len(common_files)) / float(len(union_files)) if union_files else 0.0

    expected_overlap = 0.0
    manifest_overlap = 0.0
    max_single_file = 0.0
    prod_complement = 1.0

    for f in common_files:
        p_a = float(task_a_probs[f])
        p_b = float(task_b_probs[f])
        joint_prob = p_a * p_b

        expected_overlap += joint_prob
        prod_complement *= max(0.0, 1.0 - joint_prob)

        if joint_prob > max_single_file:
            max_single_file = joint_prob

        if is_manifest_file(f):
            manifest_overlap += joint_prob

    p_any_overlap = float(max(0.0, min(1.0, 1.0 - prod_complement)))

    feature_dict = {
        "expected_overlap": float(expected_overlap),
        "p_any_overlap": p_any_overlap,
        "manifest_overlap": float(manifest_overlap),
        "max_single_file_overlap": float(max_single_file),
        "jaccard_candidate_overlap": float(jaccard),
        "issue_text_similarity": float(np.clip(issue_text_similarity, -1.0, 1.0)),
        "repo_churn_14d": float(max(0.0, repo_churn_14d)),
        "agent_heterogeneity": float(1.0 if agent_heterogeneity else 0.0),
        "duration_overlap_hours": float(max(0.0, duration_overlap_hours)),
    }

    return PairwiseOverlapFeatures(features=feature_dict)
