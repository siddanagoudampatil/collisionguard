import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from collisionguard.conflict.calibration import compute_expected_calibration_error
from collisionguard.sim.scheduler import SimulationMetrics

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray | List[int],
    y_prob: np.ndarray | List[float],
) -> Dict[str, float]:
    """Compute comprehensive classification and calibration metrics for conflict forecasting.

    Returns:
        Dictionary containing:
            - roc_auc: Area Under the Receiver Operating Characteristic curve.
            - pr_auc: Area Under the Precision-Recall curve (Average Precision).
            - ece: Expected Calibration Error across 10 probability bins.
            - brier_score: Mean squared error of probability predictions.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    if len(y_t) == 0:
        return {
            "roc_auc": 0.5,
            "pr_auc": 0.0,
            "ece": 0.0,
            "brier_score": 0.0,
        }

    # If single-class edge case, fall back safely
    if len(np.unique(y_t)) < 2:
        return {
            "roc_auc": 0.5,
            "pr_auc": float(np.mean(y_t)),
            "ece": float(np.mean(np.abs(y_t - y_p))),
            "brier_score": float(np.mean((y_t - y_p) ** 2)),
        }

    roc_auc = float(roc_auc_score(y_t, y_p))
    pr_auc = float(average_precision_score(y_t, y_p))
    ece, _, _ = compute_expected_calibration_error(y_t, y_p, n_bins=10)
    brier = float(brier_score_loss(y_t, y_p))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ece": ece,
        "brier_score": brier,
    }


def compute_pareto_frontier(results: List[SimulationMetrics]) -> List[SimulationMetrics]:
    """Extract non-dominated Pareto optimal simulation operating points.

    An operating point A dominates B if:
    - conflicts_avoided(A) >= conflicts_avoided(B) AND
    - total_delay_hours(A) <= total_delay_hours(B)
    with at least one strict inequality.
    """
    if not results:
        return []

    pareto_points: List[SimulationMetrics] = []
    for candidate in results:
        is_dominated = False
        for other in results:
            if candidate is other:
                continue
            better_or_equal_avoided = other.conflicts_avoided >= candidate.conflicts_avoided
            better_or_equal_delay = other.total_delay_hours <= candidate.total_delay_hours
            strictly_better = (
                other.conflicts_avoided > candidate.conflicts_avoided
                or other.total_delay_hours < candidate.total_delay_hours
            )

            if better_or_equal_avoided and better_or_equal_delay and strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto_points.append(candidate)

    # Sort frontier by increasing delay hours (and increasing conflicts avoided)
    pareto_points.sort(key=lambda m: (m.total_delay_hours, m.conflicts_avoided))
    return pareto_points
