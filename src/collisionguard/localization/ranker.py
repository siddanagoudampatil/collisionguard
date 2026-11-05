from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from collisionguard.localization.cochange import CochangePatternMiner, normalize_posix_path
from collisionguard.localization.retrieval import CandidateRetrievalResult, tokenize_path_and_text

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

FEATURE_COLUMNS: List[str] = [
    "bm25_score",
    "dense_sim",
    "exact_path_mention",
    "file_depth",
    "is_manifest",
    "is_test",
    "commit_frequency",
    "edit_recency_days",
    "edit_recency_decay",
    "fpgrowth_max_conf",
    "fpgrowth_max_lift",
    "file_extension_len",
    "path_char_len",
    "token_overlap_count",
    "lexical_rank",
]


@dataclass
class CandidateFeatureVector:
    """Represents the 15 tabular features computed for a candidate file."""

    file_path: str
    features: Dict[str, float]


def extract_candidate_features(
    candidate_path: str,
    issue_title: str,
    issue_body: str,
    retrieval_result: CandidateRetrievalResult,
    cochange_miner: Optional[CochangePatternMiner] = None,
    file_history_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    task_created_at: Optional[datetime | pd.Timestamp] = None,
    decay_lambda: float = 0.05,
) -> CandidateFeatureVector:
    """Compute the 15 tabular ranking features for a single candidate file."""
    norm_path = normalize_posix_path(candidate_path)
    path_obj = Path(norm_path)
    file_name = path_obj.name.lower()
    full_text = f"{issue_title} {issue_body}".lower()

    # 1. Retrieval scores
    bm25_score = float(retrieval_result.bm25_scores.get(norm_path, 0.0))
    dense_sim = float(retrieval_result.dense_similarities.get(norm_path, 0.0))

    # 2. Syntactic / Path features
    exact_mention = 1.0 if (norm_path.lower() in full_text or (len(file_name) > 3 and file_name in full_text)) else 0.0
    file_depth = float(len(path_obj.parts) - 1)
    is_manifest = 1.0 if file_name in MANIFEST_FILENAMES else 0.0
    is_test = 1.0 if any(k in norm_path.lower() for k in ("test", "spec", "__tests__")) else 0.0

    # 3. Historical activity features
    stats = (file_history_stats or {}).get(norm_path, {})
    commit_freq = float(stats.get("commit_count", 0))
    last_edit_time = stats.get("last_committed_at")

    recency_days = 365.0
    recency_decay = 0.0
    if last_edit_time and task_created_at:
        try:
            t_task = pd.to_datetime(task_created_at, utc=True)
            t_edit = pd.to_datetime(last_edit_time, utc=True)
            delta = max(0.0, (t_task - t_edit).total_seconds() / 86400.0)
            recency_days = delta
            recency_decay = math.exp(-decay_lambda * delta)
        except Exception:
            pass

    # 4. FP-Growth co-change features
    fpgrowth_conf = 0.0
    fpgrowth_lift = 0.0
    if cochange_miner:
        # Link candidate against top-5 lexical matches
        top_5_lex = sorted(
            retrieval_result.bm25_scores.keys(),
            key=lambda k: retrieval_result.bm25_scores[k],
            reverse=True,
        )[:5]
        fpgrowth_conf, fpgrowth_lift = cochange_miner.get_cochange_metrics(norm_path, top_5_lex)

    # 5. String & Token properties
    ext = path_obj.suffix.lstrip(".")
    ext_len = float(len(ext))
    path_len = float(len(norm_path))

    path_tokens = set(tokenize_path_and_text(norm_path))
    query_tokens = set(tokenize_path_and_text(f"{issue_title} {issue_body}"))
    overlap_count = float(len(path_tokens & query_tokens))

    # 6. Lexical rank position (1-indexed, 0 if unranked)
    ranked_lex = sorted(
        retrieval_result.bm25_scores.keys(),
        key=lambda k: retrieval_result.bm25_scores[k],
        reverse=True,
    )
    lex_rank = float(ranked_lex.index(norm_path) + 1) if norm_path in ranked_lex else 0.0

    feature_dict: Dict[str, float] = {
        "bm25_score": bm25_score,
        "dense_sim": dense_sim,
        "exact_path_mention": exact_mention,
        "file_depth": file_depth,
        "is_manifest": is_manifest,
        "is_test": is_test,
        "commit_frequency": commit_freq,
        "edit_recency_days": recency_days,
        "edit_recency_decay": recency_decay,
        "fpgrowth_max_conf": fpgrowth_conf,
        "fpgrowth_max_lift": fpgrowth_lift,
        "file_extension_len": ext_len,
        "path_char_len": path_len,
        "token_overlap_count": overlap_count,
        "lexical_rank": lex_rank,
    }

    return CandidateFeatureVector(file_path=norm_path, features=feature_dict)


class FootprintRanker:
    """LightGBM LambdaMART ranker predicting file touch probabilities P(f in Footprint)."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state

        self.model: Optional[lgb.LGBMRanker] = None
        self.score_mean: float = 0.0
        self.score_std: float = 1.0

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        groups: List[int],
    ) -> "FootprintRanker":
        """Train LightGBM LambdaMART ranker optimizing NDCG@10."""
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            importance_type="gain",
        )

        df_features = pd.DataFrame(X, columns=FEATURE_COLUMNS) if isinstance(X, np.ndarray) else X
        self.model.fit(df_features, y, group=groups, eval_at=[10])

        # Calibrate mean and std for sigmoid mapping on training distribution
        raw_preds = self.model.predict(df_features)
        self.score_mean = float(np.mean(raw_preds))
        self.score_std = float(np.std(raw_preds)) if np.std(raw_preds) > 1e-6 else 1.0

        logger.info(
            "Trained FootprintRanker on %d instances across %d tasks (score mean: %.3f, std: %.3f)",
            len(y),
            len(groups),
            self.score_mean,
            self.score_std,
        )
        return self

    def predict_scores(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Output raw LambdaMART ranking scores."""
        if self.model is None:
            raise RuntimeError("FootprintRanker is not fitted yet.")
        df_features = pd.DataFrame(X, columns=FEATURE_COLUMNS) if isinstance(X, np.ndarray) else X
        return self.model.predict(df_features)

    def predict_probabilities(
        self,
        X: pd.DataFrame | np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Convert ranking scores into calibrated marginal probabilities P(f in Footprint)."""
        scores = self.predict_scores(X)
        temp = max(1e-4, temperature)
        # Temperature-scaled standardized sigmoid: 1 / (1 + exp(-(s - mu) / (sigma * temp)))
        scaled = (scores - self.score_mean) / (self.score_std * temp)
        # Clip scaled scores to avoid numerical overflow in exp
        clipped = np.clip(scaled, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))
