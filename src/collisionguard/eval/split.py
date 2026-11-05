import logging
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from collisionguard.data.dataset import ConcurrentTaskPair, TaskRecord
from collisionguard.localization.ranker import FootprintRanker, extract_candidate_features
from collisionguard.localization.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class RepositoryGroupKFold:
    """Strict repository-grouped K-Fold cross-validator preventing entity leakage."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        data: List[ConcurrentTaskPair] | pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices for training and validation splits grouped strictly by repo_url.

        Guarantees that repositories in validation never appear in the training split.
        """
        if isinstance(data, pd.DataFrame):
            repos = data["repo_url"].values
            n_samples = len(data)
        else:
            repos = np.array([p.repo_url for p in data])
            n_samples = len(data)

        unique_repos = np.unique(repos)
        if len(unique_repos) < self.n_splits:
            raise ValueError(
                f"Number of unique repositories ({len(unique_repos)}) must be greater than or equal to n_splits ({self.n_splits})"
            )

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(unique_repos)
            repo_order = {repo: idx for idx, repo in enumerate(unique_repos)}
            group_codes = np.array([repo_order[r] for r in repos])
        else:
            group_codes = repos

        gkf = GroupKFold(n_splits=self.n_splits)
        dummy_X = np.zeros(n_samples)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(dummy_X, groups=group_codes)):
            train_repos: Set[str] = set(repos[train_idx])
            val_repos: Set[str] = set(repos[val_idx])

            # Enforce entity leakage invariant
            leakage = train_repos & val_repos
            if leakage:
                raise RuntimeError(f"Entity leakage detected in fold {fold}: {leakage}")

            logger.info(
                "Fold %d: %d train samples (%d repos), %d val samples (%d repos)",
                fold + 1,
                len(train_idx),
                len(train_repos),
                len(val_idx),
                len(val_repos),
            )
            yield train_idx, val_idx


def generate_out_of_fold_footprints(
    tasks: List[TaskRecord],
    repo_file_trees: Dict[str, List[str]],
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Produce out-of-fold Stage 1 candidate probability distributions P(f in Footprint_task).

    Stage 2 must ingest candidate probabilities generated on held-out repositories
    to prevent optimistic footprint overfitting.
    """
    by_repo: Dict[str, List[TaskRecord]] = {}
    for t in tasks:
        by_repo.setdefault(t.repo_url, []).append(t)

    unique_repos = np.array(sorted(list(by_repo.keys())))
    if len(unique_repos) < n_splits:
        # Fall back to self-predictions if repository count is smaller than folds
        n_splits = max(2, len(unique_repos))

    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_repos)

    splits = np.array_split(unique_repos, n_splits)
    oof_predictions: Dict[str, Dict[str, float]] = {}

    for fold_idx, val_repos in enumerate(splits):
        val_repo_set = set(val_repos)
        train_tasks = [t for r, r_tasks in by_repo.items() if r not in val_repo_set for t in r_tasks]
        val_tasks = [t for r, r_tasks in by_repo.items() if r in val_repo_set for t in r_tasks]

        if not train_tasks or not val_tasks:
            continue

        # Build Stage 1 training matrix for train_tasks
        train_feature_rows: List[List[float]] = []
        train_targets: List[int] = []
        train_groups: List[int] = []

        for task in train_tasks:
            repo_files = repo_file_trees.get(task.repo_url, [])
            if not repo_files:
                continue
            retriever = HybridRetriever(repo_files)
            retrieval_res = retriever.retrieve(task.issue_title, task.issue_body)

            modified_set = set(task.modified_files)
            task_cand_count = len(retrieval_res.candidate_files)
            if task_cand_count == 0:
                continue

            for cand_path in retrieval_res.candidate_files:
                feat = extract_candidate_features(
                    candidate_path=cand_path,
                    issue_title=task.issue_title,
                    issue_body=task.issue_body,
                    retrieval_result=retrieval_res,
                )
                train_feature_rows.append(list(feat.features.values()))
                train_targets.append(1 if cand_path in modified_set else 0)

            train_groups.append(task_cand_count)

        if not train_feature_rows:
            continue

        ranker = FootprintRanker(n_estimators=50, random_state=random_state)
        ranker.fit(np.array(train_feature_rows), np.array(train_targets), train_groups)

        # Generate out-of-fold predictions for val_tasks
        for task in val_tasks:
            repo_files = repo_file_trees.get(task.repo_url, [])
            if not repo_files:
                oof_predictions[task.task_id] = {}
                continue
            retriever = HybridRetriever(repo_files)
            retrieval_res = retriever.retrieve(task.issue_title, task.issue_body)

            val_feature_rows: List[List[float]] = []
            cand_paths: List[str] = []

            for cand_path in retrieval_res.candidate_files:
                feat = extract_candidate_features(
                    candidate_path=cand_path,
                    issue_title=task.issue_title,
                    issue_body=task.issue_body,
                    retrieval_result=retrieval_res,
                )
                val_feature_rows.append(list(feat.features.values()))
                cand_paths.append(cand_path)

            if val_feature_rows:
                probs = ranker.predict_probabilities(np.array(val_feature_rows))
                oof_predictions[task.task_id] = {
                    cand_paths[i]: float(probs[i]) for i in range(len(cand_paths))
                }
            else:
                oof_predictions[task.task_id] = {}

    return oof_predictions
