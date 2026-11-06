from datetime import datetime
import numpy as np
import pandas as pd
import pytest

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
from collisionguard.localization.ranker import FootprintRanker, extract_candidate_features
from collisionguard.localization.retrieval import HybridRetriever
from collisionguard.sim.scheduler import QueueScheduler, prepare_sim_tasks


def test_full_collisionguard_pipeline_end_to_end():
    # 1. Setup mock repository files
    repo_url = "https://github.com/mock-org/mock-auth-service"
    repo_files = [
        "src/auth.py",
        "src/jwt_handler.py",
        "src/models.py",
        "src/views.py",
        "package.json",
        "README.md",
    ]

    base_time = pd.Timestamp("2025-01-01 10:00:00", tz="UTC")

    # 2. Four synthetic tasks: Task 1 and Task 2 conflict on src/auth.py
    tasks = [
        TaskRecord(
            task_id="TASK_001",
            repo_url=repo_url,
            head_sha="sha001",
            issue_id="ISSUE_101",
            issue_title="Implement OAuth authentication",
            issue_body="Add OAuth login flow in auth handler src/auth.py",
            issue_created_at=base_time - pd.Timedelta(hours=2),
            pr_created_at=base_time,
            pr_closed_at=base_time + pd.Timedelta(hours=3),
            modified_files=["src/auth.py", "package.json"],
        ),
        TaskRecord(
            task_id="TASK_002",
            repo_url=repo_url,
            head_sha="sha002",
            issue_id="ISSUE_102",
            issue_title="Refactor user session authentication",
            issue_body="Refactor authentication session checks in auth controller src/auth.py",
            issue_created_at=base_time - pd.Timedelta(hours=1),
            pr_created_at=base_time + pd.Timedelta(hours=1),
            pr_closed_at=base_time + pd.Timedelta(hours=4),
            modified_files=["src/auth.py", "src/jwt_handler.py"],
        ),
        TaskRecord(
            task_id="TASK_003",
            repo_url=repo_url,
            head_sha="sha003",
            issue_id="ISSUE_103",
            issue_title="Update database schema models",
            issue_body="Add created_at timestamps to models table in src/models.py",
            issue_created_at=base_time - pd.Timedelta(hours=1),
            pr_created_at=base_time + pd.Timedelta(hours=1, minutes=30),
            pr_closed_at=base_time + pd.Timedelta(hours=3, minutes=30),
            modified_files=["src/models.py"],
        ),
        TaskRecord(
            task_id="TASK_004",
            repo_url=repo_url,
            head_sha="sha004",
            issue_id="ISSUE_104",
            issue_title="Improve documentation readme",
            issue_body="Update project setup guide and deployment documentation in README.md",
            issue_created_at=base_time - pd.Timedelta(hours=1),
            pr_created_at=base_time + pd.Timedelta(hours=2),
            pr_closed_at=base_time + pd.Timedelta(hours=5),
            modified_files=["README.md"],
        ),
    ]

    # 3. Step 1: Candidate retrieval and Stage 1 ranking
    retriever = HybridRetriever(file_paths=repo_files)
    task_footprints = {}

    # Train a Stage 1 Ranker on tasks
    stage1_features = []
    stage1_targets = []
    stage1_groups = []

    for task in tasks:
        res = retriever.retrieve(task.issue_title, task.issue_body)
        cands = res.candidate_files
        mod_set = set(task.modified_files)

        if cands:
            stage1_groups.append(len(cands))
            for cand in cands:
                feat = extract_candidate_features(cand, task.issue_title, task.issue_body, res)
                stage1_features.append(list(feat.features.values()))
                stage1_targets.append(1 if cand in mod_set else 0)

    ranker = FootprintRanker(n_estimators=30, random_state=42)
    ranker.fit(np.array(stage1_features), np.array(stage1_targets), stage1_groups)

    # Predict footprints
    for task in tasks:
        res = retriever.retrieve(task.issue_title, task.issue_body)
        cands = res.candidate_files
        if cands:
            feats = [
                list(extract_candidate_features(c, task.issue_title, task.issue_body, res).features.values())
                for c in cands
            ]
            probs = ranker.predict_probabilities(np.array(feats))
            task_footprints[task.task_id] = {cands[i]: float(probs[i]) for i in range(len(cands))}
        else:
            task_footprints[task.task_id] = {}

    assert "src/auth.py" in task_footprints["TASK_001"]
    assert "src/auth.py" in task_footprints["TASK_002"]

    # 4. Construct concurrent pairs
    raw_pairs = build_concurrent_pairs(tasks)
    assert len(raw_pairs) >= 3

    # Label concurrent pairs based on modified file intersections
    pairs = [
        ConcurrentTaskPair(
            pair_id=p.pair_id,
            repo_url=p.repo_url,
            task_a=p.task_a,
            task_b=p.task_b,
            overlap_duration_hours=p.overlap_duration_hours,
            base_commit=p.base_commit,
            label=1 if bool(set(p.task_a.modified_files) & set(p.task_b.modified_files)) else 0,
            conflict_files=list(set(p.task_a.modified_files) & set(p.task_b.modified_files)),
        )
        for p in raw_pairs
    ]

    # Check that TASK_001 and TASK_002 is flagged as conflict
    conflicting_pair = [p for p in pairs if {p.task_a.task_id, p.task_b.task_id} == {"TASK_001", "TASK_002"}][0]
    assert conflicting_pair.label == 1
    assert "src/auth.py" in conflicting_pair.conflict_files

    # 5. Extract Stage 2 features and train ConflictRiskClassifier
    X_rows = []
    y_labels = []
    for p in pairs:
        probs_a = task_footprints.get(p.task_a.task_id, {})
        probs_b = task_footprints.get(p.task_b.task_id, {})
        overlap_feat = compute_pairwise_overlap_features(
            probs_a,
            probs_b,
            issue_text_similarity=0.4,
            repo_churn_14d=5.0,
            agent_heterogeneity=False,
            duration_overlap_hours=float(p.overlap_duration_hours),
        )
        X_rows.append(overlap_feat.to_array())
        y_labels.append(int(p.label or 0))

    X = np.array(X_rows)
    y = np.array(y_labels)

    clf = ConflictRiskClassifier(n_estimators=30, min_child_samples=1, random_state=42)
    clf.fit(X, y)
    raw_pred_probs = clf.predict_proba(X)

    # 6. Isotonic Calibration
    calibrator = IsotonicProbabilityCalibrator()
    calibrator.fit(raw_pred_probs, y)
    calibrated_probs = calibrator.calibrate(raw_pred_probs)

    assert len(calibrated_probs) == len(pairs)
    assert np.all(calibrated_probs >= 0.0) and np.all(calibrated_probs <= 1.0)

    # Compute classification metrics
    metrics_summary = compute_classification_metrics(y, calibrated_probs)
    assert "roc_auc" in metrics_summary
    assert "pr_auc" in metrics_summary
    assert "ece" in metrics_summary

    # 7. Evaluate baselines on the pairs
    base_rate = BaseRateBaseline().fit(pairs, y)
    assert len(base_rate.predict_proba(pairs)) == len(pairs)

    bm25_base = BM25OverlapBaseline(repo_file_trees={repo_url: repo_files}).fit(pairs, y)
    assert len(bm25_base.predict_proba(pairs)) == len(pairs)

    text_base = TextSimilarityBaseline().fit(pairs, y)
    assert len(text_base.predict_proba(pairs)) == len(pairs)

    oracle = TrueFileFootprintOracle(n_estimators=20).fit(pairs, y)
    assert len(oracle.predict_proba(pairs)) == len(pairs)

    # 8. Discrete-event queue simulation using calibrated model predictions
    pairwise_dict = {}
    for idx, p in enumerate(pairs):
        pairwise_dict[(p.task_a.task_id, p.task_b.task_id)] = float(calibrated_probs[idx])
        pairwise_dict[(p.task_b.task_id, p.task_a.task_id)] = float(calibrated_probs[idx])

    scheduler = QueueScheduler(
        tau=0.25,
        pairwise_probs=pairwise_dict,
        ground_truth_conflicts={(p.task_a.task_id, p.task_b.task_id) for p in pairs if p.label == 1},
    )

    sim_metrics = scheduler.run(tasks)
    assert sim_metrics.total_tasks == 4
    # The high risk conflicting pair (TASK_001, TASK_002) is avoided by gating
    assert sim_metrics.conflicts_avoided >= 1
    assert sim_metrics.total_delay_hours >= 0.0
