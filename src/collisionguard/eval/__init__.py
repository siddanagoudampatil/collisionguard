"""Evaluation, validation splitting, baselines, interpretability, taxonomy, leakage, and metric modules."""

from collisionguard.eval.baselines import (
    BaseConflictBaseline,
    BaseRateBaseline,
    BM25OverlapBaseline,
    TextSimilarityBaseline,
    TrueFileFootprintOracle,
)
from collisionguard.eval.interpretability import ConflictRiskExplainer
from collisionguard.eval.leakage import LeakageAuditor
from collisionguard.eval.metrics import (
    compute_classification_metrics,
    compute_pareto_frontier,
)
from collisionguard.eval.split import (
    RepositoryGroupKFold,
    generate_out_of_fold_footprints,
)
from collisionguard.eval.taxonomy import (
    ConflictCategory,
    classify_pair_conflict,
    compute_taxonomy_metrics,
    parse_merge_tree_conflicts,
)

__all__ = [
    "RepositoryGroupKFold",
    "generate_out_of_fold_footprints",
    "BaseConflictBaseline",
    "BaseRateBaseline",
    "TextSimilarityBaseline",
    "BM25OverlapBaseline",
    "TrueFileFootprintOracle",
    "ConflictRiskExplainer",
    "ConflictCategory",
    "classify_pair_conflict",
    "compute_taxonomy_metrics",
    "parse_merge_tree_conflicts",
    "LeakageAuditor",
    "compute_classification_metrics",
    "compute_pareto_frontier",
]
