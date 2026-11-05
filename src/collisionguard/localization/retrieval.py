from dataclasses import dataclass, field
import logging
import re
from typing import Dict, List, Optional, Set

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def tokenize_path_and_text(text: str) -> List[str]:
    """Tokenize file paths and natural language text into normalized tokens."""
    if not text:
        return []
    # Split camelCase words
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", text)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    # Split on non-alphanumeric characters (slashes, dots, underscores, hyphens)
    tokens = re.findall(r"[a-zA-Z0-9]+", s2.lower())
    return [t for t in tokens if len(t) > 1]


@dataclass(frozen=True)
class CandidateRetrievalResult:
    """Represents the union of lexical and dense candidate files for a task."""

    candidate_files: List[str]
    bm25_scores: Dict[str, float] = field(default_factory=dict)
    dense_similarities: Dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    """Indexes repository file trees and retrieves candidate files using BM25 and bi-encoders."""

    def __init__(
        self,
        file_paths: List[str],
        file_summaries: Optional[Dict[str, str]] = None,
        model_name: str = "all-MiniLM-L6-v2",
        encoder: Optional[SentenceTransformer] = None,
    ):
        # Normalize file paths to POSIX
        self.file_paths = [p.replace("\\", "/").lstrip("/") for p in file_paths]
        self.file_summaries = file_summaries or {}

        # 1. Lexical BM25 indexing
        self.corpus_tokens: List[List[str]] = []
        for path in self.file_paths:
            summary = self.file_summaries.get(path, "")
            combined_doc = f"{path} {summary}".strip()
            self.corpus_tokens.append(tokenize_path_and_text(combined_doc))

        self.bm25: Optional[BM25Okapi] = None
        if self.corpus_tokens and any(len(t) > 0 for t in self.corpus_tokens):
            self.bm25 = BM25Okapi(self.corpus_tokens)

        # 2. Dense semantic indexing
        self.encoder = encoder or SentenceTransformer(model_name)
        if self.file_paths:
            docs_to_embed = [
                f"{p} {self.file_summaries.get(p, '')}".strip()
                for p in self.file_paths
            ]
            self.file_embeddings = self.encoder.encode(
                docs_to_embed,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:
            self.file_embeddings = np.empty((0, 384), dtype=np.float32)

    def retrieve(
        self,
        issue_title: str,
        issue_body: str,
        top_k_lexical: int = 50,
        top_k_semantic: int = 50,
        max_candidates: int = 80,
    ) -> CandidateRetrievalResult:
        """Query BM25 and dense bi-encoder, returning the union of top-k candidates capped at max_candidates."""
        if not self.file_paths:
            return CandidateRetrievalResult(candidate_files=[])

        query_text = f"{issue_title} {issue_body}".strip()
        query_tokens = tokenize_path_and_text(query_text)

        # BM25 lexical scoring
        bm25_scores_dict: Dict[str, float] = {}
        top_lexical_files: List[str] = []
        if self.bm25 and query_tokens:
            raw_bm25_scores = self.bm25.get_scores(query_tokens)
            ranked_lexical_indices = np.argsort(raw_bm25_scores)[::-1]
            k_lex = min(top_k_lexical, len(self.file_paths))
            for idx in ranked_lexical_indices[:k_lex]:
                path = self.file_paths[idx]
                score = float(raw_bm25_scores[idx])
                bm25_scores_dict[path] = score
                top_lexical_files.append(path)

        # Dense semantic scoring
        dense_sims_dict: Dict[str, float] = {}
        top_semantic_files: List[str] = []
        if len(self.file_paths) > 0 and query_text:
            query_emb = self.encoder.encode(
                [query_text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            # Cosine similarity is dot product of normalized vectors
            sims = np.dot(self.file_embeddings, query_emb)
            ranked_semantic_indices = np.argsort(sims)[::-1]
            k_sem = min(top_k_semantic, len(self.file_paths))
            for idx in ranked_semantic_indices[:k_sem]:
                path = self.file_paths[idx]
                sim = float(sims[idx])
                dense_sims_dict[path] = sim
                top_semantic_files.append(path)

        # Candidate pool is the union of lexical and semantic matches
        candidate_set: Set[str] = set(top_lexical_files) | set(top_semantic_files)

        # If union exceeds max_candidates, prioritize files with joint signals or higher combined ranks
        if len(candidate_set) > max_candidates:
            # Score each candidate by reciprocal rank in both retrieval lists
            lex_rank = {p: i + 1 for i, p in enumerate(top_lexical_files)}
            sem_rank = {p: i + 1 for i, p in enumerate(top_semantic_files)}

            def priority_key(p: str) -> float:
                score = 0.0
                if p in lex_rank:
                    score += 1.0 / lex_rank[p]
                if p in sem_rank:
                    score += 1.0 / sem_rank[p]
                return score

            sorted_candidates = sorted(list(candidate_set), key=priority_key, reverse=True)
            candidate_list = sorted(sorted_candidates[:max_candidates])
        else:
            candidate_list = sorted(list(candidate_set))

        return CandidateRetrievalResult(
            candidate_files=candidate_list,
            bm25_scores=bm25_scores_dict,
            dense_similarities=dense_sims_dict,
        )
