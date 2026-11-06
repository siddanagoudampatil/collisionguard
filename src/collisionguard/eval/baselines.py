from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from collisionguard.conflict.model import ConflictRiskClassifier
from collisionguard.conflict.overlap import compute_pairwise_overlap_features
from collisionguard.data.dataset import ConcurrentTaskPair, TaskRecord
from collisionguard.localization.retrieval import HybridRetriever, tokenize_path_and_text

logger = logging.getLogger(__name__)


def _get_pair_label(p: Any) -> int:
    """Extract binary conflict label supporting both label and is_conflict fields."""
    if hasattr(p, "label") and p.label is not None:
        return int(p.label)
    if hasattr(p, "is_conflict"):
        return 1 if p.is_conflict else 0
    return 0


def _get_pair_overlap_hours(p: Any) -> float:
    """Extract overlap duration supporting both field naming variations."""
    return float(getattr(p, "overlap_duration_hours", getattr(p, "duration_overlap_hours", 0.0)))


class BaseConflictBaseline(ABC):
    """Abstract interface for merge conflict baselines."""

    @abstractmethod
    def fit(self, pairs: List[ConcurrentTaskPair], y: Optional[np.ndarray] = None) -> "BaseConflictBaseline":
        pass

    @abstractmethod
    def predict_proba(self, pairs: List[ConcurrentTaskPair]) -> np.ndarray:
        pass

    def predict(self, pairs: List[ConcurrentTaskPair], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(pairs)
        return (probs >= threshold).astype(int)


class BaseRateBaseline(BaseConflictBaseline):
    """Predicts a constant probability matching the empirical training incidence."""

    def __init__(self, base_rate: float = 0.23):
        self.base_rate = float(base_rate)

    def fit(self, pairs: List[ConcurrentTaskPair], y: Optional[np.ndarray] = None) -> "BaseRateBaseline":
        if y is not None and len(y) > 0:
            self.base_rate = float(np.mean(y))
        elif pairs:
            labels = [_get_pair_label(p) for p in pairs]
            self.base_rate = float(np.mean(labels))
        return self

    def predict_proba(self, pairs: List[ConcurrentTaskPair]) -> np.ndarray:
        n = len(pairs)
        return np.full(n, self.base_rate, dtype=float)


class TextSimilarityBaseline(BaseConflictBaseline):
    """Logistic regression baseline trained solely on semantic cosine similarity of issue texts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embedder = None
        self.clf = LogisticRegression(random_state=42)
        self.is_fitted = False

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _compute_cosine_similarities(self, pairs: List[ConcurrentTaskPair]) -> np.ndarray:
        if not pairs:
            return np.empty((0, 1), dtype=float)

        embedder = self._get_embedder()
        # Collect unique issue texts to embed efficiently
        text_cache: Dict[str, np.ndarray] = {}
        for p in pairs:
            for task in (p.task_a, p.task_b):
                if task.task_id not in text_cache:
                    full_text = f"{task.issue_title} {task.issue_body}".strip()
                    emb = embedder.encode(full_text, convert_to_numpy=True)
                    norm = np.linalg.norm(emb)
                    text_cache[task.task_id] = emb / (norm + 1e-12)

        sims = []
        for p in pairs:
            vec_a = text_cache[p.task_a.task_id]
            vec_b = text_cache[p.task_b.task_id]
            sim = float(np.dot(vec_a, vec_b))
            sims.append(sim)

        return np.array(sims, dtype=float).reshape(-1, 1)

    def fit(self, pairs: List[ConcurrentTaskPair], y: Optional[np.ndarray] = None) -> "TextSimilarityBaseline":
        X = self._compute_cosine_similarities(pairs)
        if y is None:
            y = np.array([_get_pair_label(p) for p in pairs])

        if len(np.unique(y)) > 1:
            self.clf.fit(X, y)
        else:
            # Fallback if single class
            self.clf.classes_ = np.array([0, 1])
            self.clf.intercept_ = np.array([0.0])
            self.clf.coef_ = np.array([[1.0]])

        self.is_fitted = True
        return self

    def predict_proba(self, pairs: List[ConcurrentTaskPair]) -> np.ndarray:
        if not pairs:
            return np.empty(0, dtype=float)
        X = self._compute_cosine_similarities(pairs)
        if not self.is_fitted:
            # Simple sigmoid scaling of cosine similarity if unfitted
            return 1.0 / (1.0 + np.exp(-3.0 * X.flatten()))
        return self.clf.predict_proba(X)[:, 1]


class BM25OverlapBaseline(BaseConflictBaseline):
    """Raw lexical BM25 candidate overlap heuristic without learned footprint ranker."""

    def __init__(self, top_k: int = 50, repo_file_trees: Optional[Dict[str, List[str]]] = None):
        self.top_k = top_k
        self.repo_file_trees = repo_file_trees or {}
        self.clf = LogisticRegression(random_state=42)
        self.is_fitted = False

    def _extract_bm25_candidates(self, task: TaskRecord) -> Set[str]:
        repo_files = self.repo_file_trees.get(task.repo_url, [])
        if not repo_files:
            return set()
        corpus_tokens = [tokenize_path_and_text(p) for p in repo_files]
        if not any(corpus_tokens):
            return set()
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi(corpus_tokens)
        query_text = f"{task.issue_title} {task.issue_body}".strip()
        query_tokens = tokenize_path_and_text(query_text)
        if not query_tokens:
            return set()
        scores = bm25.get_scores(query_tokens)
        k = min(self.top_k, len(repo_files))
        ranked_idx = np.argsort(scores)[::-1][:k]
        return {repo_files[i] for i in ranked_idx if scores[i] > 0}

    def _compute_overlap_features(self, pairs: List[ConcurrentTaskPair]) -> np.ndarray:
        if not pairs:
            return np.empty((0, 1), dtype=float)

        task_candidates: Dict[str, Set[str]] = {}
        for p in pairs:
            for task in (p.task_a, p.task_b):
                if task.task_id not in task_candidates:
                    task_candidates[task.task_id] = self._extract_bm25_candidates(task)

        jaccards = []
        for p in pairs:
            cand_a = task_candidates[p.task_a.task_id]
            cand_b = task_candidates[p.task_b.task_id]
            union = cand_a | cand_b
            inter = cand_a & cand_b
            jaccard = float(len(inter)) / float(len(union)) if union else 0.0
            jaccards.append(jaccard)

        return np.array(jaccards, dtype=float).reshape(-1, 1)

    def fit(self, pairs: List[ConcurrentTaskPair], y: Optional[np.ndarray] = None) -> "BM25OverlapBaseline":
        X = self._compute_overlap_features(pairs)
        if y is None:
            y = np.array([_get_pair_label(p) for p in pairs])

        if len(np.unique(y)) > 1:
            self.clf.fit(X, y)
        else:
            self.clf.classes_ = np.array([0, 1])
            self.clf.intercept_ = np.array([0.0])
            self.clf.coef_ = np.array([[2.0]])

        self.is_fitted = True
        return self

    def predict_proba(self, pairs: List[ConcurrentTaskPair]) -> np.ndarray:
        if not pairs:
            return np.empty(0, dtype=float)
        X = self._compute_overlap_features(pairs)
        if not self.is_fitted:
            # Fallback linear clipping
            return np.clip(X.flatten(), 0.0, 1.0)
        return self.clf.predict_proba(X)[:, 1]


class TrueFileFootprintOracle(BaseConflictBaseline):
    """Theoretical upper bound: Stage 2 LightGBM evaluated with ground-truth modified files."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.classifier = ConflictRiskClassifier(n_estimators=n_estimators, random_state=random_state)
        self.is_fitted = False

    def _extract_oracle_features(self, pairs: List[ConcurrentTaskPair]) -> np.ndarray:
        feature_rows = []
        for p in pairs:
            # Perfect Stage 1 probability footprint: 1.0 for true modified files
            probs_a = {f: 1.0 for f in p.task_a.modified_files}
            probs_b = {f: 1.0 for f in p.task_b.modified_files}

            agent_a = getattr(p.task_a, "agent_id", None)
            agent_b = getattr(p.task_b, "agent_id", None)
            agent_hetero = (agent_a != agent_b) if (agent_a and agent_b) else False

            feat = compute_pairwise_overlap_features(
                probs_a,
                probs_b,
                issue_text_similarity=0.0,
                repo_churn_14d=0.0,
                agent_heterogeneity=agent_hetero,
                duration_overlap_hours=_get_pair_overlap_hours(p),
            )
            feature_rows.append(feat.to_array())

        return np.array(feature_rows, dtype=float) if feature_rows else np.empty((0, 9), dtype=float)

    def fit(self, pairs: List[ConcurrentTaskPair], y: Optional[np.ndarray] = None) -> "TrueFileFootprintOracle":
        if not pairs:
            return self
        X = self._extract_oracle_features(pairs)
        if y is None:
            y = np.array([_get_pair_label(p) for p in pairs])

        self.classifier.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, pairs: List[ConcurrentTaskPair]) -> np.ndarray:
        if not pairs:
            return np.empty(0, dtype=float)
        X = self._extract_oracle_features(pairs)
        return self.classifier.predict_proba(X)
