"""Evaluation, validation splitting, baselines, and metric modules."""

from collisionguard.eval.baselines import (
    BaseConflictBaseline,
    BaseRateBaseline,
    BM25OverlapBaseline,
    TextSimilarityBaseline,
    TrueFileFootprintOracle,
)
from collisionguard.eval.split import (
    RepositoryGroupKFold,
    generate_out_of_fold_footprints,
)

__all__ = [
    "RepositoryGroupKFold",
    "generate_out_of_fold_footprints",
    "BaseConflictBaseline",
    "BaseRateBaseline",
    "TextSimilarityBaseline",
    "BM25OverlapBaseline",
    "TrueFileFootprintOracle",
]
