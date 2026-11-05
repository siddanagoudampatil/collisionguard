from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_posix_path(path: str) -> str:
    """Normalize file path to POSIX format, stripping leading slashes and whitespace."""
    if not path:
        return ""
    clean = str(path).replace("\\", "/").strip().lstrip("/")
    # Clean possible git status prefixes like 'M\t' or 'A\t' if present
    if "\t" in clean:
        clean = clean.split("\t")[-1].strip()
    return clean


class CochangePatternMiner:
    """Mines commit co-change patterns using FP-Growth and association rules."""

    def __init__(
        self,
        min_support: float = 0.01,
        min_occurrences: int = 2,
        min_confidence: float = 0.1,
        max_files_per_commit: int = 30,
    ):
        self.min_support = min_support
        self.min_occurrences = min_occurrences
        self.min_confidence = min_confidence
        self.max_files_per_commit = max_files_per_commit

        # Lookup: antecedents (file_i) -> {consequent (file_j): (confidence, lift)}
        self.rule_lookup: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
        self.rules_df: Optional[pd.DataFrame] = None

    def build_transactions(
        self,
        commit_records: List[Dict[str, Any]] | pd.DataFrame,
    ) -> List[List[str]]:
        """Group modified files by commit, excluding bulk refactorings (> max_files_per_commit)."""
        if isinstance(commit_records, pd.DataFrame):
            records = commit_records.to_dict("records")
        else:
            records = commit_records

        by_commit: Dict[str, Set[str]] = defaultdict(set)
        for row in records:
            sha = str(row.get("commit_sha") or "")
            raw_path = row.get("file_path")
            if not sha or not raw_path:
                continue
            posix_path = normalize_posix_path(raw_path)
            if posix_path:
                by_commit[sha].add(posix_path)

        transactions: List[List[str]] = []
        for sha, files in by_commit.items():
            # Exclude bulk refactorings and single-file commits
            if 1 < len(files) <= self.max_files_per_commit:
                transactions.append(sorted(list(files)))

        logger.info(
            "Extracted %d valid transactions from %d commits",
            len(transactions),
            len(by_commit),
        )
        return transactions

    def fit(self, transactions: List[List[str]]) -> "CochangePatternMiner":
        """Mine frequent itemsets and association rules from commit transactions."""
        self.rule_lookup.clear()
        self.rules_df = None

        if not transactions or len(transactions) < self.min_occurrences:
            logger.info("Insufficient transactions (%d) for FP-Growth mining", len(transactions))
            return self

        # Calculate effective minimum support
        effective_min_support = max(
            self.min_support,
            self.min_occurrences / len(transactions),
        )

        try:
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

            frequent_itemsets = fpgrowth(
                df_encoded,
                min_support=effective_min_support,
                use_colnames=True,
            )

            if frequent_itemsets.empty:
                logger.info("FP-Growth produced zero frequent itemsets at support %.3f", effective_min_support)
                return self

            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence,
            )

            if rules.empty:
                logger.info("Zero association rules satisfied confidence threshold %.3f", self.min_confidence)
                return self

            self.rules_df = rules

            # Populate lookup table
            for _, row in rules.iterrows():
                conf = float(row["confidence"])
                lift = float(row["lift"])
                antecedents = list(row["antecedents"])
                consequents = list(row["consequents"])

                for ante in antecedents:
                    for conseq in consequents:
                        if ante == conseq:
                            continue
                        current = self.rule_lookup[ante].get(conseq)
                        if current is None or conf > current[0]:
                            self.rule_lookup[ante][conseq] = (conf, lift)

            logger.info("Mined %d association rules covering %d files", len(rules), len(self.rule_lookup))
        except Exception as exc:
            logger.warning("FP-Growth pattern mining failed: %s", exc)

        return self

    def get_cochange_metrics(
        self,
        candidate_file: str,
        reference_files: List[str],
    ) -> Tuple[float, float]:
        """Compute maximum confidence and lift linking candidate_file to reference_files."""
        candidate = normalize_posix_path(candidate_file)
        if not candidate or not reference_files:
            return 0.0, 0.0

        max_conf = 0.0
        max_lift = 0.0

        for ref in reference_files:
            ref_path = normalize_posix_path(ref)
            if not ref_path or ref_path == candidate:
                continue

            # Check candidate -> ref
            rule1 = self.rule_lookup.get(candidate, {}).get(ref_path)
            if rule1:
                conf, lift = rule1
                if conf > max_conf:
                    max_conf = conf
                if lift > max_lift:
                    max_lift = lift

            # Check ref -> candidate
            rule2 = self.rule_lookup.get(ref_path, {}).get(candidate)
            if rule2:
                conf, lift = rule2
                if conf > max_conf:
                    max_conf = conf
                if lift > max_lift:
                    max_lift = lift

        return max_conf, max_lift
