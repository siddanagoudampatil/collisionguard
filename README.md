# CollisionGuard

Pre-execution merge conflict forecasting for concurrent coding agents.

## Overview

CollisionGuard predicts three-way Git merge conflicts between concurrent tasks before code generation begins. By analyzing issue requirements and repository commit history up to task creation time ($t_0$), CollisionGuard localizes file footprints, evaluates pairwise conflict risk, and dynamically serializes high-risk tasks while executing independent tasks in parallel.

## Pipeline Architecture

1. **Data Ingestion & Merge Replay**: Stream task pairs using DuckDB columnar pushdown; evaluate ground-truth three-way merges headlessly using `git merge-tree`.
2. **Stage 1 (File Footprint Localization)**: Retrieve file candidates via BM25 and dense bi-encoder representations (`all-MiniLM-L6-v2`), augment with FP-Growth co-change rules from pre-$t_0$ history, and re-rank candidates using LightGBM LambdaMART.
3. **Stage 2 (Pairwise Risk Estimation)**: Compute expected file overlap and manifest collision features, enforce monotonic constraints via LightGBM, and calibrate probabilities with isotonic regression.
4. **Queue Simulation**: Simulate chronological task execution queues in SimPy to quantify conflict reduction rate versus scheduling latency tradeoffs.

## Installation

```bash
git clone https://github.com/siddanagoudampatil/CollisionGuard.git
cd CollisionGuard
pip install -e .
```

## Running Tests

```bash
pytest tests/
```
