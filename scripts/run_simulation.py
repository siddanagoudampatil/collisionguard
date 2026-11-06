#!/usr/bin/env python3
"""CollisionGuard SimPy Queue Simulation and Pareto Sweep CLI Runner.

Simulates concurrent pull request execution under threshold-gated queue scheduling,
evaluates conflicts avoided versus cumulative queue delay hours across tau sweeps,
and outputs the Pareto efficiency frontier to CSV.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from collisionguard.data.dataset import TaskRecord
from collisionguard.eval.metrics import compute_pareto_frontier
from collisionguard.sim.scheduler import QueueScheduler, SimTask, SimulationMetrics, prepare_sim_tasks, simulate_sweep

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_simulation_workload(n_tasks: int = 40, seed: int = 42) -> Tuple[List[SimTask], Dict[Tuple[str, str], float]]:
    """Synthesize concurrent task streams with calibrated conflict probabilities."""
    rng = np.random.default_rng(seed)
    files_pool = [f"src/module_{i}.py" for i in range(15)]
    tasks: List[SimTask] = []

    current_time = 0.0
    for i in range(n_tasks):
        # Inter-arrival Poisson process: exponential delay
        current_time += rng.exponential(scale=1.5)
        duration = float(np.clip(rng.lognormal(mean=1.2, sigma=0.5), 0.5, 8.0))

        # File modifications
        k_files = rng.integers(1, 4)
        chosen_files = list(rng.choice(files_pool, size=k_files, replace=False))

        tasks.append(
            SimTask(
                task_id=f"PR_{i+1:03d}",
                arrival_time=current_time,
                duration_hours=duration,
                modified_files=chosen_files,
                repo_url="https://github.com/org/repo-sim",
            )
        )

    # Precompute pairwise calibrated probabilities
    pairwise_probs: Dict[Tuple[str, str], float] = {}
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            t_a = tasks[i]
            t_b = tasks[j]
            overlap = set(t_a.modified_files) & set(t_b.modified_files)
            if overlap:
                # Strong predicted risk when genuine file overlap exists
                p = float(rng.uniform(0.65, 0.98))
            else:
                p = float(rng.uniform(0.01, 0.25))

            pairwise_probs[(t_a.task_id, t_b.task_id)] = p
            pairwise_probs[(t_b.task_id, t_a.task_id)] = p

    return tasks, pairwise_probs


def main():
    parser = argparse.ArgumentParser(description="CollisionGuard SimPy Queue Simulation CLI")
    parser.add_argument("--eval-pairs", type=str, default="", help="Path to evaluation tasks/pairs Parquet")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save Pareto tradeoff CSV")
    parser.add_argument("--tau-min", type=float, default=0.0, help="Minimum threshold for tau sweep")
    parser.add_argument("--tau-max", type=float, default=1.0, help="Maximum threshold for tau sweep")
    parser.add_argument("--tau-step", type=float, default=0.05, help="Step size for tau sweep")
    parser.add_argument("--seed", type=int, default=42, help="Simulation random seed")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing SimPy discrete-event simulation (tau: %.2f -> %.2f, step: %.2f)...", args.tau_min, args.tau_max, args.tau_step)

    tasks, pairwise_probs = generate_simulation_workload(n_tasks=50, seed=args.seed)
    tau_values = [round(x, 2) for x in np.arange(args.tau_min, args.tau_max + 1e-6, args.tau_step)]

    sweep_results = simulate_sweep(
        tasks=tasks,
        tau_values=tau_values,
        pairwise_probs=pairwise_probs,
    )

    pareto_frontier = compute_pareto_frontier(sweep_results)
    pareto_taus = {p.tau for p in pareto_frontier}

    rows = []
    for r in sweep_results:
        d = r.to_dict()
        d["is_pareto_optimal"] = r.tau in pareto_taus
        rows.append(d)

    df_results = pd.DataFrame(rows)
    csv_out = output_path / "pareto_tradeoff_curve.csv"
    df_results.to_csv(csv_out, index=False)

    print("\n" + "=" * 80)
    print("SIMPY QUEUE SCHEDULING SIMULATION: PARETO TRADEOFF RESULTS")
    print("=" * 80)
    display_cols = [
        "tau",
        "conflicts_avoided",
        "unmitigated_conflicts",
        "conflicts_avoided_pct",
        "false_delay_incidents",
        "total_delay_hours",
        "mean_delay_hours",
        "is_pareto_optimal",
    ]
    print(df_results[display_cols].to_string(index=False))
    print("=" * 80)
    logger.info("Pareto tradeoff frontier written to %s", csv_out)


if __name__ == "__main__":
    main()
