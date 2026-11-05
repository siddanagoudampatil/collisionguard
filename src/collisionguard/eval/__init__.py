"""Evaluation, validation splitting, and metric modules."""

from collisionguard.eval.split import (
    RepositoryGroupKFold,
    generate_out_of_fold_footprints,
)

__all__ = [
    "RepositoryGroupKFold",
    "generate_out_of_fold_footprints",
]
