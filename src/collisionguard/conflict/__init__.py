"""Stage 2: Pairwise conflict risk modeling modules."""

from collisionguard.conflict.overlap import (
    STAGE2_FEATURE_COLUMNS,
    PairwiseOverlapFeatures,
    compute_pairwise_overlap_features,
)

__all__ = [
    "STAGE2_FEATURE_COLUMNS",
    "PairwiseOverlapFeatures",
    "compute_pairwise_overlap_features",
]
