import numpy as np
import pandas as pd
import pytest

from collisionguard.conflict.model import ConflictRiskClassifier
from collisionguard.conflict.overlap import STAGE2_FEATURE_COLUMNS
from collisionguard.data.dataset import ConcurrentTaskPair, TaskRecord
from collisionguard.eval.interpretability import ConflictRiskExplainer
from collisionguard.eval.leakage import LeakageAuditor
from collisionguard.eval.taxonomy import (
    ConflictCategory,
    classify_pair_conflict,
    compute_taxonomy_metrics,
    parse_merge_tree_conflicts,
)


def test_shap_tree_explainer_dimensions():
    rng = np.random.default_rng(42)
    n_samples = 25
    n_features = len(STAGE2_FEATURE_COLUMNS)

    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features))
    y = rng.choice([0, 1], size=n_samples)

    clf = ConflictRiskClassifier(n_estimators=15, min_child_samples=2, random_state=42)
    clf.fit(X, y)

    explainer = ConflictRiskExplainer(clf)
    shap_matrix = explainer.explain(X)

    # Check output shape
    assert shap_matrix.shape == (n_samples, n_features)
    assert np.all(np.isfinite(shap_matrix))

    # Mean absolute attributions
    attributions = explainer.compute_mean_abs_attributions(X)
    assert len(attributions) == n_features
    for col in STAGE2_FEATURE_COLUMNS:
        assert col in attributions
        assert attributions[col] >= 0.0

    # Top features
    top_3 = explainer.get_top_features(X, top_k=3)
    assert len(top_3) == 3
    # First top feature score must be >= second top feature score
    assert top_3[0][1] >= top_3[1][1]


def test_conflict_taxonomy_parser_modern():
    mock_modern_stdout = """
Auto-merging src/collisionguard/conflict/model.py
CONFLICT (content): Merge conflict in src/collisionguard/conflict/model.py
CONFLICT (modify/delete): config/settings.json deleted in HEAD and modified in branch-b
CONFLICT (add/add): Merge conflict in tests/test_new_feature.py
"""
    results = parse_merge_tree_conflicts(mock_modern_stdout)
    assert len(results) == 3

    categories = [cat for cat, _ in results]
    files = [f for _, f in results]

    assert ConflictCategory.CONTENT in categories
    assert ConflictCategory.MODIFY_DELETE in categories
    assert ConflictCategory.ADD_ADD in categories

    assert "src/collisionguard/conflict/model.py" in files
    assert "config/settings.json" in files
    assert "tests/test_new_feature.py" in files


def test_conflict_taxonomy_parser_legacy():
    mock_legacy_stdout = """
changed in both
  base   100644 1111111111111111111111111111111111111111 src/app.py
  our    100644 2222222222222222222222222222222222222222 src/app.py
  their  100644 3333333333333333333333333333333333333333 src/app.py
added in both
  our    100644 4444444444444444444444444444444444444444 schema.sql
  their  100644 5555555555555555555555555555555555555555 schema.sql
"""
    results = parse_merge_tree_conflicts(mock_legacy_stdout)
    assert len(results) == 2

    cat_map = dict(results)
    assert cat_map.get(ConflictCategory.CONTENT) == "src/app.py"
    assert cat_map.get(ConflictCategory.ADD_ADD) == "schema.sql"


def test_taxonomy_metrics_computation():
    y_true_cats = [
        ConflictCategory.CONTENT,
        ConflictCategory.CONTENT,
        ConflictCategory.MODIFY_DELETE,
        ConflictCategory.ADD_ADD,
        ConflictCategory.CLEAN,
    ]
    y_prob = [0.95, 0.85, 0.75, 0.65, 0.10]

    metrics = compute_taxonomy_metrics(y_true_cats, y_prob, threshold=0.5)

    assert "content" in metrics
    assert "modify_delete" in metrics
    assert "add_add" in metrics

    assert metrics["content"]["support"] == 2
    assert metrics["modify_delete"]["support"] == 1
    assert metrics["add_add"]["support"] == 1

    for cat_name, m in metrics.items():
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["pr_auc"] <= 1.0


def test_leakage_auditor_delta_checks():
    auditor = LeakageAuditor(random_state=42)
    base_t = pd.Timestamp("2025-01-01 10:00:00", tz="UTC")

    task = TaskRecord(
        task_id="T_AUDIT_1",
        repo_url="https://github.com/org/repo-audit",
        head_sha="sha1",
        issue_id="ISSUE_1",
        issue_title="Update authentication handler",
        issue_body="Refactor authentication session checks",
        issue_created_at=base_t - pd.Timedelta(hours=2),
        pr_created_at=base_t,
        pr_closed_at=base_t + pd.Timedelta(hours=2),
        modified_files=["src/auth.py", "src/tokens.py"],
    )

    c_title, c_body = auditor.contaminate_task(
        task,
        diff_summary="@@ -1,5 +1,5 @@ auth handler updates",
        commit_messages=["fix auth logic"],
    )

    # Contaminated text must leak actual modified file paths and diff tokens
    assert "src/auth.py" in c_title or "src/auth.py" in c_body
    assert "auth handler updates" in c_body

    # Audit retrieval hit rate on small repo
    repo_file_trees = {"https://github.com/org/repo-audit": ["src/auth.py", "src/tokens.py", "README.md"]}
    hit_report = auditor.audit_retrieval_hit_rate([task], repo_file_trees, top_k=10)

    assert "clean_hit_at_k" in hit_report
    assert "contaminated_hit_at_k" in hit_report
    assert hit_report["contaminated_hit_at_k"] >= hit_report["clean_hit_at_k"]
