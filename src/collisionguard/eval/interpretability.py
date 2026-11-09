import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap

from collisionguard.conflict.model import ConflictRiskClassifier
from collisionguard.conflict.overlap import STAGE2_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class ConflictRiskExplainer:
    """Tree-based SHAP feature attribution explainer for Stage 2 conflict risk models."""

    def __init__(
        self,
        model: Union[ConflictRiskClassifier, Any],
        feature_names: Optional[List[str]] = None,
    ):
        """Initialize SHAP TreeExplainer for LightGBM conflict classifier.

        Args:
            model: ConflictRiskClassifier or underlying LightGBM model.
            feature_names: Feature column names corresponding to feature matrix columns.
        """
        self.feature_names = feature_names or STAGE2_FEATURE_COLUMNS

        # Extract underlying LightGBM booster or classifier if wrapped
        if isinstance(model, ConflictRiskClassifier):
            if model.model is None:
                raise RuntimeError("ConflictRiskClassifier must be fitted before initializing explainer.")
            self.raw_model = model.model
        else:
            self.raw_model = model

        self.explainer = shap.TreeExplainer(self.raw_model)

    def explain(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute SHAP attribution values for input samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray of SHAP values of shape (n_samples, n_features) for positive conflict class.
        """
        df_features = (
            pd.DataFrame(X, columns=self.feature_names)
            if isinstance(X, np.ndarray)
            else X
        )

        raw_shap_values = self.explainer.shap_values(df_features)

        # Handle version differences in SHAP outputs for binary LightGBM
        if isinstance(raw_shap_values, list):
            # Binary classification returns [class_0, class_1] in some versions
            shap_matrix = (
                raw_shap_values[1] if len(raw_shap_values) > 1 else raw_shap_values[0]
            )
        elif isinstance(raw_shap_values, np.ndarray):
            if raw_shap_values.ndim == 3:
                # Shape (n_samples, n_features, n_classes)
                shap_matrix = raw_shap_values[:, :, 1]
            else:
                shap_matrix = raw_shap_values
        else:
            shap_matrix = np.asarray(raw_shap_values)

        return np.asarray(shap_matrix, dtype=float)

    def compute_mean_abs_attributions(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, float]:
        """Compute mean absolute SHAP value per feature, sorted in descending order of importance.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary mapping feature name to mean absolute SHAP score.
        """
        shap_matrix = self.explain(X)
        if len(shap_matrix) == 0:
            return {f: 0.0 for f in self.feature_names}

        mean_abs_scores = float(np.mean(np.abs(shap_matrix), axis=0)) if shap_matrix.ndim == 1 else np.mean(np.abs(shap_matrix), axis=0)
        mean_abs_arr = np.atleast_1d(mean_abs_scores)

        attributions: Dict[str, float] = {}
        for idx, feat_name in enumerate(self.feature_names):
            score = float(mean_abs_arr[idx]) if idx < len(mean_abs_arr) else 0.0
            attributions[feat_name] = round(score, 6)

        # Sort descending by importance
        sorted_attributions = dict(
            sorted(attributions.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_attributions

    def get_top_features(
        self, X: Union[pd.DataFrame, np.ndarray], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return top-k most influential features by mean absolute SHAP attribution."""
        attributions = self.compute_mean_abs_attributions(X)
        return list(attributions.items())[:top_k]

    def export_attributions(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        output_path: Union[str, Path],
        file_format: str = "json",
    ) -> Path:
        """Save feature attribution summary to structured JSON or CSV file.

        Args:
            X: Feature matrix.
            output_path: Destination file path.
            file_format: Output format ('json' or 'csv').

        Returns:
            Path object to written file.
        """
        attributions = self.compute_mean_abs_attributions(X)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if file_format.lower() == "csv":
            df = pd.DataFrame(
                list(attributions.items()),
                columns=["feature_name", "mean_abs_shap"],
            )
            df.to_csv(path, index=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(attributions, f, indent=2)

        logger.info("Exported SHAP feature attributions to %s", path)
        return path
