import pandas as pd
import pytest

from collisionguard.localization.cochange import CochangePatternMiner
from collisionguard.localization.retrieval import HybridRetriever, tokenize_path_and_text


def test_tokenize_path_and_text():
    tokens = tokenize_path_and_text("src/auth/UserAuthenticator.py Fix login timeout")
    assert "src" in tokens
    assert "auth" in tokens
    assert "user" in tokens
    assert "authenticator" in tokens
    assert "login" in tokens
    assert "timeout" in tokens


def test_retrieval_small_repo():
    repo_files = ["src/auth.py", "src/models/user.py", "tests/test_auth.py"]
    retriever = HybridRetriever(repo_files)
    res = retriever.retrieve("Fix login failure", "User cannot login")

    assert set(res.candidate_files) == set(repo_files)
    assert len(res.candidate_files) == 3
    assert "src/auth.py" in res.bm25_scores or "src/auth.py" in res.dense_similarities


def test_retrieval_union_bounds_large_repo():
    # Construct 120 synthetic repo files
    files = [f"src/module_{i}/service_{j}.py" for i in range(12) for j in range(10)]
    retriever = HybridRetriever(files)

    res = retriever.retrieve("Update service_3 in module_2", "service_3 database query timeout", top_k_lexical=50, top_k_semantic=50)

    # Union of top-50 lexical and top-50 semantic should be between 50 and 80 unique files
    assert 50 <= len(res.candidate_files) <= 80
    assert "src/module_2/service_3.py" in res.candidate_files


def test_retrieval_empty_repo():
    retriever = HybridRetriever([])
    res = retriever.retrieve("Fix bug", "details")
    assert res.candidate_files == []


def test_fpgrowth_transaction_filtering_and_rule_mining():
    miner = CochangePatternMiner(min_support=0.01, min_occurrences=2, min_confidence=0.1, max_files_per_commit=5)

    # Commit transactions
    records = [
        # Normal co-change commit (sha1)
        {"commit_sha": "c1", "file_path": "src/auth.py"},
        {"commit_sha": "c1", "file_path": "tests/test_auth.py"},
        {"commit_sha": "c1", "file_path": "config/auth.json"},
        # Another co-change commit (sha2)
        {"commit_sha": "c2", "file_path": "src/auth.py"},
        {"commit_sha": "c2", "file_path": "tests/test_auth.py"},
        # Bulk refactoring commit (> 5 files) -> should be excluded
        {"commit_sha": "c3", "file_path": "f1.py"},
        {"commit_sha": "c3", "file_path": "f2.py"},
        {"commit_sha": "c3", "file_path": "f3.py"},
        {"commit_sha": "c3", "file_path": "f4.py"},
        {"commit_sha": "c3", "file_path": "f5.py"},
        {"commit_sha": "c3", "file_path": "f6.py"},
        # Single-file commit -> should be excluded
        {"commit_sha": "c4", "file_path": "solo.py"},
    ]

    txs = miner.build_transactions(records)
    # c3 (bulk) and c4 (solo) excluded; only c1 and c2 remain
    assert len(txs) == 2
    assert "src/auth.py" in txs[0]

    miner.fit(txs)
    assert len(miner.rule_lookup) > 0

    conf, lift = miner.get_cochange_metrics("src/auth.py", ["tests/test_auth.py"])
    assert conf > 0.0
    assert lift > 0.0

    # Non-existent file
    conf_zero, lift_zero = miner.get_cochange_metrics("non_existent.py", ["src/auth.py"])
    assert conf_zero == 0.0
    assert lift_zero == 0.0


def test_extract_candidate_features():
    from collisionguard.localization.ranker import (
        FEATURE_COLUMNS,
        extract_candidate_features,
    )
    from collisionguard.localization.retrieval import CandidateRetrievalResult

    retrieval_res = CandidateRetrievalResult(
        candidate_files=["package.json", "tests/test_parser.py", "src/parser.py"],
        bm25_scores={"src/parser.py": 12.5, "package.json": 0.5},
        dense_similarities={"src/parser.py": 0.75, "package.json": 0.1},
    )

    feat = extract_candidate_features(
        candidate_path="package.json",
        issue_title="Update package.json dependencies",
        issue_body="Bump version and install packages",
        retrieval_result=retrieval_res,
    )

    assert len(feat.features) == 15
    assert len(feat.features) == len(FEATURE_COLUMNS)
    assert feat.features["is_manifest"] == 1.0
    assert feat.features["exact_path_mention"] == 1.0

    test_feat = extract_candidate_features(
        candidate_path="tests/test_parser.py",
        issue_title="Bug in parser",
        issue_body="Parser error",
        retrieval_result=retrieval_res,
    )
    assert test_feat.features["is_test"] == 1.0
    assert test_feat.features["is_manifest"] == 0.0


def test_footprint_ranker_fit_and_predict():
    import numpy as np
    from collisionguard.localization.ranker import FEATURE_COLUMNS, FootprintRanker

    # Synthetic dataset: 2 queries with 5 candidates each
    np.random.seed(42)
    n_samples = 10
    X = np.random.randn(n_samples, len(FEATURE_COLUMNS))
    # Binary relevance target: 1 if ground-truth modified, 0 otherwise
    y = np.array([1, 0, 0, 0, 0, 1, 1, 0, 0, 0], dtype=np.int32)
    groups = [5, 5]

    ranker = FootprintRanker(n_estimators=10, random_state=42)
    ranker.fit(X, y, groups)

    scores = ranker.predict_scores(X)
    assert len(scores) == n_samples

    probs = ranker.predict_probabilities(X)
    assert len(probs) == n_samples
    assert np.all((probs >= 0.0) & (probs <= 1.0))

