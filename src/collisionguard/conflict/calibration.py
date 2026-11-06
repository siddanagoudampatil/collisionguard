import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


def compute_expected_calibration_error(
    y_true: np.ndarray | List[int],
    y_prob: np.ndarray | List[float],
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE) and reliability diagram data.

    ECE partitions predictions into M equal-width bins and computes the weighted
    absolute difference between empirical accuracy and mean predicted confidence:
        ECE = ∑_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted conflict probabilities in [0, 1].
        n_bins: Number of equal-width bins (default: 10).

    Returns:
        Tuple of:
            - ece: Expected Calibration Error as a float in [0, 1].
            - bin_accuracies: Empirical positive rate in each bin (shape: (n_bins,)).
            - bin_confidences: Average predicted confidence in each bin (shape: (n_bins,)).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"y_true (length {len(y_true_arr)}) and y_prob (length {len(y_prob_arr)}) must have identical lengths."
        )

    n_samples = len(y_true_arr)
    if n_samples == 0:
        return 0.0, np.zeros(n_bins, dtype=float), np.zeros(n_bins, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = np.zeros(n_bins, dtype=float)
    bin_confidences = np.zeros(n_bins, dtype=float)

    # Bin assignment:
    # We assign probabilities to bins [edge_i, edge_{i+1}). The right edge 1.0 is included in the last bin.
    bin_indices = np.digitize(y_prob_arr, bin_edges) - 1
    # Handle exact 1.0 falling into bin index n_bins
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            acc = float(np.mean(y_true_arr[mask]))
            conf = float(np.mean(y_prob_arr[mask]))
            bin_accuracies[b] = acc
            bin_confidences[b] = conf
            ece += (count / n_samples) * abs(acc - conf)
        else:
            bin_confidences[b] = float((bin_edges[b] + bin_edges[b + 1]) / 2.0)
            bin_accuracies[b] = 0.0

    return float(ece), bin_accuracies, bin_confidences


class IsotonicProbabilityCalibrator:
    """Non-parametric isotonic regression calibrator for conflict risk probabilities.

    Transforms uncalibrated model scores into empirical probabilities while preserving
    monotonic ordering and handling out-of-bounds inputs gracefully.
    """

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0):
        self.y_min = y_min
        self.y_max = y_max
        self.regressor = IsotonicRegression(
            y_min=self.y_min,
            y_max=self.y_max,
            out_of_bounds="clip",
            increasing=True,
        )
        self.is_fitted: bool = False

    def fit(
        self,
        y_prob: np.ndarray | List[float],
        y_true: np.ndarray | List[int],
    ) -> "IsotonicProbabilityCalibrator":
        """Fit the isotonic regression model on out-of-fold validation scores.

        Args:
            y_prob: Raw or uncalibrated predicted probabilities from Stage 2 classifier.
            y_true: True binary merge conflict outcomes (0 or 1).
        """
        probs = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
        labels = np.asarray(y_true, dtype=float)

        if len(probs) != len(labels):
            raise ValueError("y_prob and y_true must have the same length.")

        if len(probs) == 0:
            raise ValueError("Cannot fit calibrator on empty data.")

        self.regressor.fit(probs, labels)
        self.is_fitted = True
        logger.info("Fitted IsotonicProbabilityCalibrator on %d validation samples.", len(labels))
        return self

    def calibrate(self, y_prob: np.ndarray | List[float]) -> np.ndarray:
        """Calibrate predicted probabilities using fitted isotonic regression.

        Args:
            y_prob: Raw probabilities to calibrate.

        Returns:
            Calibrated probabilities clipped to [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("IsotonicProbabilityCalibrator must be fitted before calibration.")

        probs = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
        calibrated = self.regressor.predict(probs)
        return np.clip(calibrated, self.y_min, self.y_max)

    def compute_ece(
        self,
        y_prob: np.ndarray | List[float],
        y_true: np.ndarray | List[int],
        n_bins: int = 10,
    ) -> float:
        """Compute the expected calibration error for calibrated probabilities."""
        calibrated = self.calibrate(y_prob)
        ece, _, _ = compute_expected_calibration_error(y_true, calibrated, n_bins=n_bins)
        return ece
