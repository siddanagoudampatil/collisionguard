import logging
from typing import List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from collisionguard.conflict.overlap import STAGE2_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Monotonic constraints: +1 enforces ∂P(Conflict)/∂x ≥ 0 for overlap features
# Features:
# 1. expected_overlap -> +1
# 2. p_any_overlap -> +1
# 3. manifest_overlap -> +1
# 4. max_single_file_overlap -> +1
# 5. jaccard_candidate_overlap -> +1
# 6. issue_text_similarity -> 0
# 7. repo_churn_14d -> 0
# 8. agent_heterogeneity -> 0
# 9. duration_overlap_hours -> 0
MONOTONE_CONSTRAINTS: List[int] = [1, 1, 1, 1, 1, 0, 0, 0, 0]


class ConflictRiskClassifier:
    """Monotonically constrained LightGBM classifier for pairwise merge conflict forecasting."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state

        self.model: Optional[lgb.LGBMClassifier] = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> "ConflictRiskClassifier":
        """Fit binary classifier enforcing positive monotonicity on predicted overlap metrics."""
        self.model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            monotone_constraints=MONOTONE_CONSTRAINTS,
            importance_type="gain",
        )

        df_features = pd.DataFrame(X, columns=STAGE2_FEATURE_COLUMNS) if isinstance(X, np.ndarray) else X
        self.model.fit(df_features, y)

        logger.info(
            "Trained ConflictRiskClassifier on %d pairs with monotonic constraints %s",
            len(y),
            MONOTONE_CONSTRAINTS,
        )
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict probability of 3-way merge conflict P(Conflict | A, B)."""
        if self.model is None:
            raise RuntimeError("ConflictRiskClassifier is not fitted yet.")
        df_features = pd.DataFrame(X, columns=STAGE2_FEATURE_COLUMNS) if isinstance(X, np.ndarray) else X
        # Return probability of positive class (label = 1)
        return self.model.predict_proba(df_features)[:, 1]

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary conflict label based on decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
