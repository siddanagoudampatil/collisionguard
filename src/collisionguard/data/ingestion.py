import logging
from pathlib import Path
from typing import Generator, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def create_duckdb_session(memory_limit: str = "1GB") -> duckdb.DuckDBPyConnection:
    """Initialize an in-memory DuckDB session configured with a strict memory cap."""
    conn = duckdb.connect(database=":memory:")
    conn.execute(f"PRAGMA max_memory='{memory_limit}';")
    conn.execute("PRAGMA threads=4;")
    return conn


def stream_linked_task_records(
    pr_parquet_path: str | Path,
    issue_parquet_path: str | Path,
    related_issue_parquet_path: str | Path,
    batch_size: int = 1000,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Stream linked pull request and issue records using columnar projection pushdown.

    Only retrieves tasks where issue creation timestamp is less than or equal to PR creation,
    preventing post-execution data leakage at the query level.
    """
    should_close = False
    if conn is None:
        conn = create_duckdb_session()
        should_close = True

    query = f"""
    SELECT 
        pr.id AS pr_id,
        pr.repo_url,
        pr.created_at AS pr_created_at,
        pr.closed_at AS pr_closed_at,
        pr.head_sha,
        iss.id AS issue_id,
        iss.title AS issue_title,
        iss.body AS issue_body,
        iss.created_at AS issue_created_at
    FROM '{pr_parquet_path}' pr
    JOIN '{related_issue_parquet_path}' rel ON pr.id = rel.pr_id
    JOIN '{issue_parquet_path}' iss ON rel.issue_id = iss.id
    WHERE iss.created_at <= pr.created_at
    ORDER BY pr.repo_url, pr.created_at ASC
    """

    try:
        cursor = conn.execute(query)
        while True:
            chunk = cursor.fetch_df_chunk(batch_size)
            if chunk.empty:
                break
            logger.info("Yielded task record chunk with %d records", len(chunk))
            yield chunk
    finally:
        if should_close:
            conn.close()


def stream_commit_records(
    commits_parquet_path: str | Path,
    repo_url: Optional[str] = None,
    before_timestamp: Optional[str] = None,
    batch_size: int = 2000,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Stream historical commit touch records using projected column queries."""
    should_close = False
    if conn is None:
        conn = create_duckdb_session()
        should_close = True

    filters = []
    if repo_url:
        filters.append(f"repo_url = '{repo_url}'")
    if before_timestamp:
        filters.append(f"committed_at < '{before_timestamp}'")

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    query = f"""
    SELECT 
        repo_url,
        commit_sha,
        file_path,
        committed_at
    FROM '{commits_parquet_path}'
    {where_clause}
    ORDER BY committed_at ASC
    """

    try:
        cursor = conn.execute(query)
        while True:
            chunk = cursor.fetch_df_chunk(batch_size)
            if chunk.empty:
                break
            yield chunk
    finally:
        if should_close:
            conn.close()
