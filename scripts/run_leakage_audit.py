#!/usr/bin/env python3
"""CollisionGuard Feature Leakage Audit CLI Runner.

Quantifies artificial performance inflation when post-execution text (pull request
diff summaries, commit messages, modified paths) is injected into pre-execution queries,
demonstrating strict temporal and information leakage isolation in the production pipeline.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure repository root and scripts directory are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collisionguard.data.dataset import ConcurrentTaskPair, TaskRecord, build_concurrent_pairs
from collisionguard.eval.leakage import LeakageAuditor
from run_pipeline import generate_synthetic_benchmark_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CollisionGuard Feature Leakage Audit Runner")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save leakage audit results")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    parser.add_argument("--top-k", type=int, default=50, help="Candidate retrieval top-k cutoff")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing CollisionGuard Leakage Audit (seed=%d, top_k=%d)...", args.seed, args.top_k)

    tasks, repo_file_trees = generate_synthetic_benchmark_tasks()
    raw_pairs = build_concurrent_pairs(tasks, max_pairs_per_repo=40)

    # Label pairs with ground truth modified file intersections
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

    auditor = LeakageAuditor(random_state=args.seed)

    # 1. Retrieval leakage audit
    retrieval_report = auditor.audit_retrieval_hit_rate(tasks, repo_file_trees, top_k=args.top_k)

    # 2. Pipeline conflict prediction leakage audit
    pipeline_report = auditor.audit_pipeline_leakage(pairs, repo_file_trees)

    combined_results = {
        "retrieval_hit_rate": retrieval_report,
        "conflict_prediction": pipeline_report,
    }

    # Print summary table
    print("\n" + "=" * 76)
    print("COLLISIONGUARD FEATURE LEAKAGE AUDIT REPORT")
    print("=" * 76)
    print(f"Stage 1 Clean Hit@{args.top_k}:        {retrieval_report['clean_hit_at_k']:.4f}")
    print(f"Stage 1 Contaminated Hit@{args.top_k}: {retrieval_report['contaminated_hit_at_k']:.4f}")
    print(f"Stage 1 Hit@{args.top_k} Delta:        {retrieval_report['hit_at_k_delta']:+.4f} (Artificial Inflation)")
    print("-" * 76)
    print(f"Stage 2 Clean PR-AUC:          {pipeline_report['clean_pr_auc']:.4f}")
    print(f"Stage 2 Contaminated PR-AUC:   {pipeline_report['contaminated_pr_auc']:.4f}")
    print(f"Stage 2 PR-AUC Delta:          {pipeline_report['pr_auc_inflation_delta']:+.4f} (Post-hoc Inflation)")
    print(f"Stage 2 Clean ROC-AUC:         {pipeline_report['clean_roc_auc']:.4f}")
    print(f"Stage 2 Contaminated ROC-AUC:  {pipeline_report['contaminated_roc_auc']:.4f}")
    print(f"Stage 2 ROC-AUC Delta:         {pipeline_report['roc_auc_inflation_delta']:+.4f}")
    print("=" * 76)

    status_str = "PASS (Pre-execution isolation verified)" if pipeline_report["zero_leakage_verified"] else "FAIL"
    print(f"Zero Information Leakage Invariant: [{status_str}]")
    print("=" * 76 + "\n")

    out_file = output_path / "leakage_audit_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)
    logger.info("Saved leakage audit report to %s", out_file)


if __name__ == "__main__":
    main()
