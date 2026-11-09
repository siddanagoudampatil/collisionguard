# CollisionGuard

[![CI](https://github.com/siddanagoudampatil/collisionguard/actions/workflows/ci.yml/badge.svg)](https://github.com/siddanagoudampatil/collisionguard/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)

**CollisionGuard** is a pre-execution merge conflict forecasting and discrete-event queue scheduling engine for concurrent coding agents. By analyzing issue specifications and repository commit history strictly prior to task creation ($t_0$), CollisionGuard localizes file edit footprints, evaluates pairwise conflict probabilities with monotonic risk guarantees, and schedules concurrent tasks to minimize merge failures while maximizing parallel throughput.

---

## Architecture Overview

```
                      +---------------------------------------+
                      | Raw GitHub Issues & Pull Requests     |
                      +---------------------------------------+
                                          |
                      +---------------------------------------+
                      | Stage 1: File Footprint Localization   |
                      |  - BM25Okapi + MiniLM Semantic Retrieval
                      |  - FP-Growth Co-change Mining (pre-t0)|
                      |  - LambdaMART NDCG@10 File Re-ranker  |
                      +---------------------------------------+
                                          | Candidate Footprint P(f in F)
                      +---------------------------------------+
                      | Stage 2: Pairwise Conflict Modeling   |
                      |  - Expected File & Manifest Overlap   |
                      |  - Monotonically Constrained LightGBM |
                      |  - Out-of-Fold Isotonic Calibration   |
                      +---------------------------------------+
                                          | Calibrated P(Conflict | A, B)
                      +---------------------------------------+
                      | SimPy Discrete-Event Queue Scheduling |
                      |  - Threshold Gating (tau in [0.0, 1.0])|
                      |  - Pareto Tradeoff Optimization        |
                      +---------------------------------------+
```

1. **Headless Replay & Ingestion (`src/collisionguard/data/`)**: Streams historical task records through DuckDB with memory limits, validates strict temporal validity ($T_{\text{issue}} \le T_{\text{open}}$), and verifies ground truth merge conflicts headlessly via `git merge-tree`.
2. **Stage 1: File Footprint Localization (`src/collisionguard/localization/`)**: Blends lexical (BM25) and dense bi-encoder (`all-MiniLM-L6-v2`) retrieval, extracts 15-dimensional candidate features (including FP-Growth association rules mined strictly from pre-$t_0$ commit history), and re-ranks candidates using LightGBM LambdaMART.
3. **Stage 2: Pairwise Conflict Risk & Calibration (`src/collisionguard/conflict/`)**: Computes expected file overlap $E[|F_A \cap F_B|]$, collision probability $P(\text{overlap} \ge 1)$, and build manifest risk; trains a LightGBM classifier with positive monotonic constraints ($\partial P(\text{Conflict})/\partial \text{overlap} \ge 0$), and calibrates probabilities via out-of-fold isotonic regression.
4. **Evaluation & Empirical Baselines (`src/collisionguard/eval/`)**: Benchmarks against base-rate, text cosine similarity, BM25 candidate overlap, and ground-truth file oracle baselines across 5-fold `RepositoryGroupKFold` splits. Provides SHAP tree interpretability and structural conflict taxonomy breakdowns (Content, Modify/Delete, Add/Add).
5. **Discrete-Event Queue Simulation (`src/collisionguard/sim/`)**: Models chronological PR arrivals in SimPy; dynamically serializes high-risk task pairs ($\hat{p} \ge \tau$) while executing safe tasks in parallel, mapping the Pareto efficiency frontier of conflicts avoided versus queue delay.

---

## Installation

### Prerequisites
- Python 3.10 or 3.11
- Git 2.34+

```bash
git clone https://github.com/siddanagoudampatil/collisionguard.git
cd collisionguard
pip install -e ".[dev]"
```

---

## Quickstart & CLI Execution

### 1. Run 5-Fold Repository GroupKFold Evaluation Pipeline
Trains Stage 1 LambdaMART, Stage 2 Monotonic LightGBM, and Isotonic Calibration across repository-isolated folds and evaluates against empirical baselines:

```bash
python scripts/run_pipeline.py --output-dir results --seed 42 --n-splits 5
```

### 2. Run SimPy Queue Simulation & Pareto Frontier Sweep
Simulates concurrent PR execution queues across threshold sweep $\tau \in [0.0, 1.0]$, contrasting the learned scheduler against matched random holding:

```bash
python scripts/run_simulation.py --output-dir results --tau-min 0.0 --tau-max 1.0 --tau-step 0.05
```

### 3. Run Pre-Execution Information Leakage Audit
Proves zero information leakage by artificially injecting post-execution artifacts (diffs, commit logs, modified filenames) and quantifying metric inflation:

```bash
python scripts/run_leakage_audit.py --output-dir results --top-k 50
```

### 4. Generate Publication-Quality Figures
Generates vector PDF and 300 DPI PNG figures for ACM conference manuscripts:

```bash
python scripts/plot_results.py --output-dir reports/figures
```

Generated plots:
- `fig_pareto_tradeoff.pdf`: Conflicts Avoided (%) vs. Added Task Delay Hours across threshold $\tau$.
- `fig_calibration_reliability.pdf`: Reliability diagram comparing uncalibrated vs. isotonic calibrated LightGBM against the diagonal.
- `fig_evaluation_benchmarks.pdf`: Stage 1 Hit@K candidate retrieval curves and Stage 2 PR-AUC curves.

---

## Running Unit Tests

Run the complete test suite (36 tests) across all modules:

```bash
pytest tests/ -v
```

---

## Project Structure

```
collisionguard/
├── .github/workflows/ci.yml       # GitHub Actions CI workflow (Python 3.11)
├── pyproject.toml                 # Package dependencies and build configuration
├── README.md                      # Architecture and reproduction documentation
├── reports/figures/               # Publication figures (PDF and PNG)
├── scripts/
│   ├── run_pipeline.py            # 5-fold cross-validation benchmarking runner
│   ├── run_simulation.py          # SimPy queue scheduling sweep runner
│   ├── run_leakage_audit.py       # Information leakage audit harness
│   └── plot_results.py            # ACM conference publication plotting engine
├── src/collisionguard/
│   ├── data/                      # Ingestion, temporal validation, git merge-tree
│   ├── localization/              # BM25, bi-encoder, FP-Growth, LambdaMART ranker
│   ├── conflict/                  # Overlap features, monotonic classifier, calibration
│   ├── eval/                      # GroupKFold, baselines, metrics, SHAP, taxonomy, leakage
│   └── sim/                       # SimPy discrete-event queue scheduler and Pareto sweeps
└── tests/                         # Comprehensive unit and integration test suite
```

---

## License

Apache-2.0 License. See `LICENSE` for details.
