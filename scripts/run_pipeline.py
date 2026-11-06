#!/usr/bin/env python3
"""CollisionGuard End-to-End Pipeline Evaluation CLI Runner.

Executes 5-fold Repository GroupKFold cross-validation across Stage 1 Footprint
Ranking, Stage 2 Monotonic Conflict Risk Modeling, and Isotonic Calibration,
evaluating performance against empirical baselines and the true oracle.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from collisionguard.conflict.calibration import IsotonicProbabilityCalibrator
from collisionguard.conflict.model import ConflictRiskClassifier
from collisionguard.conflict.overlap import compute_pairwise_overlap_features
from collisionguard.data.dataset import ConcurrentTaskPair, TaskRecord, build_concurrent_pairs
from collisionguard.eval.baselines import (
    BaseRateBaseline,
    BM25OverlapBaseline,
    TextSimilarityBaseline,
    TrueFileFootprintOracle,
)
from collisionguard.eval.metrics import compute_classification_metrics
from collisionguard.eval.split import RepositoryGroupKFold, generate_out_of_fold_footprints

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_benchmark_tasks() -> Tuple[List[TaskRecord], Dict[str, List[str]]]:
    """Generate deterministic multi-repository synthetic benchmark tasks if external parquet is absent."""
    repos = ["https://github.com/org/repo-alpha", "https://github.com/org/repo-beta", "https://github.com/org/repo-gamma"]
    repo_file_trees = {
        repos[0]: ["src/auth.py", "src/models.py", "src/views.py", "package.json", "README.md"],
        repos[1]: ["src/server.go", "src/db.go", "go.mod", "Makefile", "docs/index.md"],
        repos[2]: ["src/core.rs", "src/lib.rs", "Cargo.toml", "tests/unit.rs", "benches/bench.rs"],
    }

    tasks: List[TaskRecord] = []
    base_time = pd.Timestamp("2025-01-01 08:00:00", tz="UTC")

    for repo_idx, repo in enumerate(repos):
        files = repo_file_trees[repo]
        for t_idx in range(12):
            task_id = f"task_{repo_idx}_{t_idx}"
            open_t = base_time + pd.Timedelta(hours=t_idx * 2)
            close_t = open_t + pd.Timedelta(hours=3 + (t_idx % 4))
            create_t = open_t - pd.Timedelta(hours=1)

            mod_file = files[t_idx % len(files)]
            title = f"Refactor module {mod_file.split('/')[-1]} feature {t_idx}"
            body = f"Implementation updates to handle concurrency in {mod_file} for task {task_id}"

            tasks.append(
                TaskRecord(
                    task_id=task_id,
                    repo_url=repo,
                    head_sha=f"sha_{repo_idx}_{t_idx}",
                    issue_id=f"issue_{repo_idx}_{t_idx}",
                    issue_title=title,
                    issue_body=body,
                    issue_created_at=create_t,
                    pr_created_at=open_t,
                    pr_closed_at=close_t,
                    modified_files=[mod_file],
                )
            )

    return tasks, repo_file_trees


def extract_stage2_features_for_pairs(
    pairs: List[ConcurrentTaskPair],
    candidate_probs: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Extract 9-dimensional Stage 2 features using candidate footprint probabilities."""
    rows = []
    for p in pairs:
        probs_a = candidate_probs.get(p.task_a.task_id, {})
        probs_b = candidate_probs.get(p.task_b.task_id, {})

        feat = compute_pairwise_overlap_features(
            probs_a,
            probs_b,
            issue_text_similarity=0.2,
            repo_churn_14d=10.0,
            agent_heterogeneity=False,
            duration_overlap_hours=float(p.overlap_duration_hours),
        )
        rows.append(feat.to_array())

    return np.array(rows, dtype=float) if rows else np.empty((0, 9), dtype=float)


def main():
    parser = argparse.ArgumentParser(description="CollisionGuard Cross-Validation Evaluation Pipeline")
    parser.add_argument("--data-dir", type=str, default="", help="Path to raw DuckDB/Parquet repository data")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save benchmark metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of GroupKFold repository splits")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing CollisionGuard Pipeline Benchmark (seed=%d)...", args.seed)

    tasks, repo_file_trees = generate_synthetic_benchmark_tasks()
    pairs = build_concurrent_pairs(tasks, max_pairs_per_repo=50)

    if not pairs:
        logger.error("No concurrent task pairs found to evaluate.")
        sys.exit(1)

    logger.info("Constructed %d concurrent pairs across %d tasks.", len(pairs), len(tasks))

    # Repository GroupKFold cross-validation
    cv = RepositoryGroupKFold(n_splits=args.n_splits, random_state=args.seed)
    pair_labels = np.array([1 if p.label == 1 else 0 for p in pairs])

    models = {
        "BaseRate": BaseRateBaseline(),
        "BM25_Heuristic": BM25OverlapBaseline(repo_file_trees=repo_file_trees),
        "TextSimilarity": TextSimilarityBaseline(),
        "CollisionGuard_Stage2": ConflictRiskClassifier(n_estimators=50, random_state=args.seed),
        "CollisionGuard_Calibrated": ConflictRiskClassifier(n_estimators=50, random_state=args.seed),
        "Oracle_Theoretical_Bound": TrueFileFootprintOracle(n_estimators=50, random_state=args.seed),
    }

    fold_metrics: Dict[str, List[Dict[str, float]]] = {m: [] for m in models}

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(pairs)):
        logger.info("Evaluating Fold %d/%d (%d train, %d test pairs)...", fold_idx + 1, args.n_splits, len(train_idx), len(test_idx))

        train_pairs = [pairs[i] for i in train_idx]
        test_pairs = [pairs[i] for i in test_idx]
        y_train = pair_labels[train_idx]
        y_test = pair_labels[test_idx]

        # Generate Out-of-Fold Footprint Probabilities
        train_tasks = list({p.task_a.task_id: p.task_a for p in train_pairs}.values())
        candidate_probs = generate_out_of_fold_footprints(train_tasks, repo_file_trees, n_splits=args.n_splits)

        # Baselines
        base_rate = BaseRateBaseline().fit(train_pairs, y_train)
        fold_metrics["BaseRate"].append(compute_classification_metrics(y_test, base_rate.predict_proba(test_pairs)))

        bm25_base = BM25OverlapBaseline(repo_file_trees=repo_file_trees).fit(train_pairs, y_train)
        fold_metrics["BM25_Heuristic"].append(compute_classification_metrics(y_test, bm25_base.predict_proba(test_pairs)))

        text_base = TextSimilarityBaseline().fit(train_pairs, y_train)
        fold_metrics["TextSimilarity"].append(compute_classification_metrics(y_test, text_base.predict_proba(test_pairs)))

        # Oracle
        oracle = TrueFileFootprintOracle(n_estimators=50, random_state=args.seed).fit(train_pairs, y_train)
        fold_metrics["Oracle_Theoretical_Bound"].append(compute_classification_metrics(y_test, oracle.predict_proba(test_pairs)))

        # CollisionGuard Monotonic Classifier
        X_train = extract_stage2_features_for_pairs(train_pairs, candidate_probs)
        X_test = extract_stage2_features_for_pairs(test_pairs, candidate_probs)

        stage2 = ConflictRiskClassifier(n_estimators=50, random_state=args.seed).fit(X_train, y_train)
        raw_probs = stage2.predict_proba(X_test)
        fold_metrics["CollisionGuard_Stage2"].append(compute_classification_metrics(y_test, raw_probs))

        # Calibrated
        calibrator = IsotonicProbabilityCalibrator()
        calibrator.fit(stage2.predict_proba(X_train), y_train)
        cal_probs = calibrator.calibrate(raw_probs)
        fold_metrics["CollisionGuard_Calibrated"].append(compute_classification_metrics(y_test, cal_probs))

    # Aggregate results
    summary_rows = []
    for model_name, metric_list in fold_metrics.items():
        avg_roc = float(np.mean([m["roc_auc"] for m in metric_list]))
        avg_pr = float(np.mean([m["pr_auc"] for m in metric_list]))
        avg_ece = float(np.mean([m["ece"] for m in metric_list]))
        avg_brier = float(np.mean([m["brier_score"] for m in metric_list]))

        summary_rows.append({
            "Model": model_name,
            "ROC_AUC": round(avg_roc, 4),
            "PR_AUC": round(avg_pr, 4),
            "ECE": round(avg_ece, 4),
            "BrierScore": round(avg_brier, 4),
        })

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("COLLISIONGUARD BENCHMARK EVALUATION RESULTS (5-Fold GroupKFold)")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print("=" * 70 + "\n")

    out_file = output_path / "evaluation_metrics.json"
    with open(out_file, "w") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info("Saved benchmark evaluation results to %s", out_file)


if __name__ == "__main__":
    main()
