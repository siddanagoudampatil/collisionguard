import numpy as np
import pytest

from collisionguard.conflict.calibration import (
    IsotonicProbabilityCalibrator,
    compute_expected_calibration_error,
)
from collisionguard.conflict.model import (
    MONOTONE_CONSTRAINTS,
    ConflictRiskClassifier,
)
from collisionguard.conflict.overlap import (
    STAGE2_FEATURE_COLUMNS,
    compute_pairwise_overlap_features,
    is_manifest_file,
)
from collisionguard.eval.split import RepositoryGroupKFold


def test_is_manifest_file():
    assert is_manifest_file("package.json")
    assert is_manifest_file("package-lock.json")
    assert is_manifest_file("yarn.lock")
    assert is_manifest_file("frontend/package.json")
    assert is_manifest_file("Cargo.toml")
    assert is_manifest_file("Cargo.lock")
    assert is_manifest_file("setup.py")
    assert is_manifest_file("pyproject.toml")
    assert is_manifest_file("go.mod")
    assert is_manifest_file("requirements.txt")
    assert is_manifest_file("nested/path/pom.xml")
    assert not is_manifest_file("src/main.py")
    assert not is_manifest_file("README.md")


def test_pairwise_overlap_identical_prob_one_returns_exact_count():
    files = [f"src/module_{i}.py" for i in range(5)]
    probs_a = {f: 1.0 for f in files}
    probs_b = {f: 1.0 for f in files}

    features = compute_pairwise_overlap_features(probs_a, probs_b)
    feat_dict = features.features

    assert pytest.approx(feat_dict["expected_overlap"]) == 5.0
    assert pytest.approx(feat_dict["p_any_overlap"]) == 1.0
    assert pytest.approx(feat_dict["jaccard_candidate_overlap"]) == 1.0
    assert pytest.approx(feat_dict["max_single_file_overlap"]) == 1.0


def test_pairwise_overlap_disjoint_candidates():
    probs_a = {"src/a.py": 0.8, "src/b.py": 0.6}
    probs_b = {"src/c.py": 0.9, "src/d.py": 0.5}

    features = compute_pairwise_overlap_features(
        probs_a,
        probs_b,
        issue_text_similarity=0.45,
        repo_churn_14d=12,
        agent_heterogeneity=True,
        duration_overlap_hours=3.5,
    )
    feat_dict = features.features

    assert feat_dict["expected_overlap"] == 0.0
    assert feat_dict["p_any_overlap"] == 0.0
    assert feat_dict["manifest_overlap"] == 0.0
    assert feat_dict["max_single_file_overlap"] == 0.0
    assert feat_dict["jaccard_candidate_overlap"] == 0.0
    assert feat_dict["issue_text_similarity"] == 0.45
    assert feat_dict["repo_churn_14d"] == 12.0
    assert feat_dict["agent_heterogeneity"] == 1.0
    assert feat_dict["duration_overlap_hours"] == 3.5


def test_pairwise_overlap_identical_and_manifest():
    probs_a = {
        "src/common.py": 0.8,
        "package.json": 0.5,
    }
    probs_b = {
        "src/common.py": 0.5,
        "package.json": 0.4,
    }

    features = compute_pairwise_overlap_features(
        probs_a,
        probs_b,
        issue_text_similarity=0.9,
        repo_churn_14d=5,
        agent_heterogeneity=False,
        duration_overlap_hours=1.0,
    )
    feat_dict = features.features

    expected_common = 0.8 * 0.5
    expected_manifest = 0.5 * 0.4
    expected_total = expected_common + expected_manifest

    assert pytest.approx(feat_dict["expected_overlap"], rel=1e-5) == expected_total
    assert pytest.approx(feat_dict["manifest_overlap"], rel=1e-5) == expected_manifest
    assert pytest.approx(feat_dict["max_single_file_overlap"], rel=1e-5) == 0.4  # 0.8 * 0.5

    # p_any_overlap = 1 - (1 - 0.4) * (1 - 0.20) = 1 - 0.6 * 0.8 = 1 - 0.48 = 0.52
    assert pytest.approx(feat_dict["p_any_overlap"], rel=1e-5) == 0.52
    assert pytest.approx(feat_dict["jaccard_candidate_overlap"], rel=1e-5) == 1.0


def test_pairwise_overlap_edge_cases_nan_and_empty():
    empty_features = compute_pairwise_overlap_features({}, {})
    for col in STAGE2_FEATURE_COLUMNS:
        assert np.isfinite(empty_features.features[col])
        assert empty_features.features[col] >= 0.0

    nan_probs_a = {"src/foo.py": float("nan"), "src/bar.py": float("inf")}
    nan_probs_b = {"src/foo.py": 0.5, "src/bar.py": -0.2}
    sanitized_features = compute_pairwise_overlap_features(nan_probs_a, nan_probs_b)
    for col in STAGE2_FEATURE_COLUMNS:
        val = sanitized_features.features[col]
        assert np.isfinite(val)
        assert not np.isnan(val)


def test_monotonic_risk_guarantee():
    rng = np.random.default_rng(42)
    n_samples = 200

    # Generate synthetic training dataset respecting monotonic intuition
    X_rows = []
    y_labels = []

    for _ in range(n_samples):
        exp_overlap = rng.uniform(0.0, 5.0)
        p_any = float(min(1.0, exp_overlap / 2.0 + rng.uniform(-0.1, 0.1)))
        p_any = max(0.0, min(1.0, p_any))
        manifest_overlap = rng.uniform(0.0, 1.0)
        max_single = rng.uniform(0.0, 1.0)
        jaccard = rng.uniform(0.0, 1.0)
        sim = rng.uniform(-0.2, 0.9)
        churn = rng.uniform(0.0, 50.0)
        hetero = float(rng.choice([0.0, 1.0]))
        dur = rng.uniform(0.1, 24.0)

        # Conflict probability strongly correlates with overlap
        true_latent = 0.8 * exp_overlap + 1.2 * p_any + 0.5 * manifest_overlap - 1.5
        prob_conflict = 1.0 / (1.0 + np.exp(-true_latent))
        y = 1 if rng.uniform(0.0, 1.0) < prob_conflict else 0

        X_rows.append([
            exp_overlap,
            p_any,
            manifest_overlap,
            max_single,
            jaccard,
            sim,
            churn,
            hetero,
            dur,
        ])
        y_labels.append(y)

    X = np.array(X_rows)
    y = np.array(y_labels)

    clf = ConflictRiskClassifier(n_estimators=60, min_child_samples=5, random_state=42)
    clf.fit(X, y)

    # Test Monotonicity: Vary expected_overlap from 0.0 to 5.0 keeping other features fixed
    test_grid_size = 30
    eval_matrix = np.tile(X[0], (test_grid_size, 1))
    overlap_values = np.linspace(0.0, 5.0, test_grid_size)
    eval_matrix[:, 0] = overlap_values  # Set expected_overlap feature

    preds = clf.predict_proba(eval_matrix)

    # Every successive predicted probability must be >= previous (non-decreasing)
    diffs = np.diff(preds)
    # Allow tiny numerical tolerance for floating-point precision
    assert np.all(diffs >= -1e-6), f"Monotonicity violated: minimum difference was {np.min(diffs)}"


def test_ece_computation_and_reliability():
    # Perfect calibration test
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.05, 0.15, 0.15, 0.2, 0.85, 0.85, 0.9, 0.95, 0.95])

    ece, bin_acc, bin_conf = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 0.25
    assert len(bin_acc) == 5
    assert len(bin_conf) == 5

    # Pathological miscalibration test (predict 1.0 for true label 0)
    y_true_bad = np.array([0, 0, 0, 0, 0])
    y_prob_bad = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    ece_bad, _, _ = compute_expected_calibration_error(y_true_bad, y_prob_bad, n_bins=10)
    assert pytest.approx(ece_bad, rel=1e-3) == 1.0


def test_isotonic_probability_calibrator():
    rng = np.random.default_rng(42)
    uncalibrated_probs = np.array([0.1, 0.2, 0.35, 0.45, 0.6, 0.75, 0.85, 0.95])
    true_labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])

    calibrator = IsotonicProbabilityCalibrator()
    calibrator.fit(uncalibrated_probs, true_labels)

    test_probs = np.array([0.05, 0.3, 0.5, 0.8, 1.0])
    calibrated_preds = calibrator.calibrate(test_probs)

    assert len(calibrated_preds) == len(test_probs)
    assert np.all(calibrated_preds >= 0.0)
    assert np.all(calibrated_preds <= 1.0)
    # Output must be non-decreasing
    assert np.all(np.diff(calibrated_preds) >= 0.0)


def test_repository_groupkfold_split_no_leakage():
    class DummyPair:
        def __init__(self, repo_url: str):
            self.repo_url = repo_url

    pairs = [
        DummyPair("repo/alpha"),
        DummyPair("repo/alpha"),
        DummyPair("repo/beta"),
        DummyPair("repo/beta"),
        DummyPair("repo/gamma"),
        DummyPair("repo/gamma"),
        DummyPair("repo/delta"),
        DummyPair("repo/epsilon"),
    ]

    cv = RepositoryGroupKFold(n_splits=3, shuffle=False)
    splits = list(cv.split(pairs))

    assert len(splits) == 3
    for train_idx, val_idx in splits:
        train_repos = {pairs[i].repo_url for i in train_idx}
        val_repos = {pairs[i].repo_url for i in val_idx}
        # Invariant: Disjoint repositories between train and validation
        assert len(train_repos & val_repos) == 0
