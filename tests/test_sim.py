import numpy as np
import pytest

from collisionguard.eval.metrics import compute_pareto_frontier
from collisionguard.sim.scheduler import (
    QueueScheduler,
    SimTask,
    SimulationMetrics,
    prepare_sim_tasks,
    simulate_random_holding_baseline,
    simulate_sweep,
)


def test_queue_scheduler_holding_and_dispatch():
    # Task 1 and Task 2 both touch "core.py" (conflict)
    # Task 3 touches "docs.md" (disjoint)
    t1 = SimTask(task_id="T1", arrival_time=0.0, duration_hours=2.0, modified_files=["core.py"])
    t2 = SimTask(task_id="T2", arrival_time=0.5, duration_hours=2.0, modified_files=["core.py"])
    t3 = SimTask(task_id="T3", arrival_time=1.0, duration_hours=1.0, modified_files=["docs.md"])

    # Threshold tau=0.5: T2 should be held until T1 finishes at t=2.0
    scheduler = QueueScheduler(tau=0.5)
    metrics = scheduler.run([t1, t2, t3])

    assert metrics.total_tasks == 3
    assert metrics.conflicts_avoided == 1
    assert metrics.unmitigated_conflicts == 0
    assert metrics.false_delay_incidents == 0
    assert pytest.approx(metrics.total_delay_hours, rel=1e-3) == 1.5  # T2 waited 2.0 - 0.5 = 1.5h
    assert pytest.approx(metrics.mean_delay_hours, rel=1e-3) == 0.5  # 1.5 / 3 tasks
    assert pytest.approx(metrics.max_delay_hours, rel=1e-3) == 1.5
    assert metrics.conflicts_avoided_pct == 100.0


def test_queue_scheduler_zero_delay_at_tau_1():
    t1 = SimTask(task_id="T1", arrival_time=0.0, duration_hours=2.0, modified_files=["core.py"])
    t2 = SimTask(task_id="T2", arrival_time=0.5, duration_hours=2.0, modified_files=["core.py"])

    # High threshold tau=1.0: no holding
    # Prob is 1.0, but if pairwise prob is 0.99 or tau > prob, no delay
    scheduler = QueueScheduler(tau=1.0, pairwise_probs={("T1", "T2"): 0.95})
    metrics = scheduler.run([t1, t2])

    assert metrics.total_delay_hours == 0.0
    assert metrics.conflicts_avoided == 0
    assert metrics.unmitigated_conflicts == 1
    assert metrics.conflicts_avoided_pct == 0.0


def test_queue_scheduler_empty_stream():
    scheduler = QueueScheduler(tau=0.5)
    metrics = scheduler.run([])
    assert metrics.total_tasks == 0
    assert metrics.total_delay_hours == 0.0


def test_simulate_sweep_and_pareto_frontier():
    t1 = SimTask(task_id="T1", arrival_time=0.0, duration_hours=3.0, modified_files=["app.py"])
    t2 = SimTask(task_id="T2", arrival_time=1.0, duration_hours=3.0, modified_files=["app.py"])
    t3 = SimTask(task_id="T3", arrival_time=1.5, duration_hours=2.0, modified_files=["app.py"])

    pairwise_probs = {
        ("T1", "T2"): 0.85,
        ("T1", "T3"): 0.75,
        ("T2", "T3"): 0.65,
    }

    sweep_results = simulate_sweep(
        tasks=[t1, t2, t3],
        tau_values=[0.1, 0.5, 0.7, 0.9],
        pairwise_probs=pairwise_probs,
    )
    assert len(sweep_results) == 4

    # Delay should decrease or stay flat as tau increases
    delays = [r.total_delay_hours for r in sweep_results]
    assert delays[0] >= delays[-1]

    frontier = compute_pareto_frontier(sweep_results)
    assert len(frontier) > 0
    # Every frontier point must have non-negative metrics
    for p in frontier:
        assert p.conflicts_avoided >= 0
        assert p.total_delay_hours >= 0.0


def test_simulate_random_holding_baseline():
    dummy_results = [
        SimulationMetrics(
            tau=0.5,
            total_tasks=10,
            unmitigated_conflicts=1,
            conflicts_avoided=4,
            false_delay_incidents=1,
            total_delay_hours=15.0,
            mean_delay_hours=1.5,
            max_delay_hours=3.0,
            conflicts_avoided_pct=80.0,
        ),
        SimulationMetrics(
            tau=1.0,
            total_tasks=10,
            unmitigated_conflicts=5,
            conflicts_avoided=0,
            false_delay_incidents=0,
            total_delay_hours=0.0,
            mean_delay_hours=0.0,
            max_delay_hours=0.0,
            conflicts_avoided_pct=0.0,
        ),
    ]

    random_metrics = simulate_random_holding_baseline([], dummy_results)
    assert len(random_metrics) == 2
    assert random_metrics[1].total_delay_hours == 0.0
    assert random_metrics[1].conflicts_avoided == 0
    assert random_metrics[0].total_delay_hours == 15.0
