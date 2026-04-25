#!/usr/bin/env python3
"""
Circle Packing Evolution - Single-Task Optimization.

Problem: Pack N circles inside a unit square [0,1]x[0,1] to maximize sum of radii.

Optimizes code using GEPA's RefinerConfig for per-evaluation refinement.
Matches batch runner pattern for a single problem size (N=26).
"""

import argparse
import json
import os
from typing import Any, Optional

import numpy as np


from utils import (
    execute_code,
    SEED_CODE,
)
from llms import (
    CIRCLE_PACKING_BACKGROUND,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    RefinerConfig,
    optimize_anything,
)


# Constants
NUM_CIRCLES = 26
LLM_MODEL = "openai/gpt-5.1"
TIMEOUT = 600


class StateTracker:
    """Track best solution and metric calls for logging."""

    def __init__(self, log_dir: str, max_metric_calls: int):
        self.max_metric_calls = max_metric_calls
        self.metric_calls = 0
        self.best_score = 0.0
        self.best_solution = None
        self.best_artifact = None
        self.logs = []
        self.log_dir = log_dir

    def record(
        self,
        score: Optional[float] = None,
        solution: Optional[Any] = None,
        artifact: Optional[Any] = None,
    ) -> None:
        self.metric_calls += 1

        if score is not None and solution is not None:
            if score > self.best_score:
                self.best_score = score
                self.best_solution = solution
                self.best_artifact = artifact
                print(f"New best solution found: {score:.4f}")

        self.log_state()

        if self.metric_calls >= self.max_metric_calls:
            print("Max metric calls reached!")

    def get_best_solution(self) -> tuple[float, Any]:
        return self.best_score, self.best_solution

    def log_state(self) -> None:
        print(f"Logging state... Best score: {self.best_score:.4f}")
        log = {
            "metric_calls": self.metric_calls,
            "best_score": self.best_score,
            "best_solution": json.dumps(self.best_solution.tolist())
            if self.best_solution is not None
            else None,
        }
        if self.best_artifact is not None:
            for key, value in self.best_artifact.items():
                if isinstance(value, np.ndarray):
                    log[f"best_artifact_{key}"] = json.dumps(value.tolist())
                else:
                    log[f"best_artifact_{key}"] = value
        self.logs.append(log)
        self.save_logs()

    def save_logs(self) -> None:
        with open(os.path.join(self.log_dir, "state_tracker_logs.json"), "w") as f:
            json.dump(self.logs, f, indent=2)


def compute_multiple_metrics(all_scores: list[float]) -> dict[str, float]:
    alpha_fixed = 0.1
    ema_fixed = all_scores[0]
    for s in all_scores[1:]:
        ema_fixed = alpha_fixed * s + (1 - alpha_fixed) * ema_fixed
    alpha_adaptive = 2.0 / (len(all_scores) + 1)
    ema_adaptive = all_scores[0]
    for s in all_scores[1:]:
        ema_adaptive = alpha_adaptive * s + (1 - alpha_adaptive) * ema_adaptive
    return {
        "max_score": max(all_scores),
        "mean_score": sum(all_scores) / len(all_scores),
        "ema_score_fixed": ema_fixed,
        "ema_score_adaptive": ema_adaptive,
    }


def create_fitness_function(state_tracker, timeout=TIMEOUT, run_logger=None):
    """Create fitness function for single-task evaluation.
    Refinement is handled by GEPA's RefinerConfig."""

    def fitness_fn(candidate, **kwargs):
        """Evaluate code candidate on N=26 circle packing."""
        code = candidate["code"]

        _, global_best_solution = state_tracker.get_best_solution()

        result = execute_code(code, timeout, global_best_solution, num_circles=NUM_CIRCLES)

        if result["success"]:
            circles = result["result"]["circles"]
            score = result["result"]["validation_details"]["sum_radii"]
            metrics = compute_multiple_metrics(result["result"]["all_scores"])
        else:
            circles = None
            score = 0.0
            metrics = None

        side_info = {
            "scores": {"sum_radii": score},
            "metrics": metrics,
            "code": code,
            "circles": circles,
            "num_circles": NUM_CIRCLES,
            "stdout": result.get("stdout", ""),
            "error": result.get("error"),
            "traceback": result.get("traceback"),
            "validation_details": result.get("result", {}).get("validation_details"),
        }

        state_tracker.record(
            score=score,
            solution=circles,
            artifact={"code": code},
        )
        if run_logger is not None:
            run_logger.record_evaluation(
                problem_id=NUM_CIRCLES,
                candidate_kind="code",
                score=score,
                success=result["success"],
                metric_calls=state_tracker.metric_calls,
                best_scores={NUM_CIRCLES: state_tracker.best_score},
                error=None if result["success"] else result.get("error"),
                evaluation_time_seconds=result.get("execution_time"),
            )

        return score, side_info

    return fitness_fn


def main():
    parser = argparse.ArgumentParser(description="Circle Packing Single-Task Optimization")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-metric-calls", type=int, default=150)
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    parser.add_argument("--model", type=str, default=LLM_MODEL)
    args = parser.parse_args()

    max_metric_calls = args.max_metric_calls
    timeout = args.timeout
    model = args.model

    # log_dir defaults to ./logs (co-located with main.py) so GEPA resumes from
    # the bundled gepa_state.bin. Override with --run-name <name> to write a fresh run.
    if args.run_name:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", args.run_name)
    else:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("Circle Packing Evolution - Single-Task Optimization")
    print("=" * 70)
    print(f"LLM Model: {model}")
    print(f"Problem size: N={NUM_CIRCLES}")
    print(f"Max metric calls: {max_metric_calls}")
    print(f"Timeout: {timeout}")
    print(f"Log directory: {log_dir}")
    print("=" * 70 + "\n")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    state_tracker = StateTracker(log_dir=log_dir, max_metric_calls=max_metric_calls)
    run_logger = None  # ExperimentRunLogger stripped from artifact bundle (cost-tracking only)

    seed_candidate = {"code": SEED_CODE}

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_metric_calls=max_metric_calls,
            track_best_outputs=True,
            cache_evaluation=True,
            frontier_type="objective",
        ),
        reflection=ReflectionConfig(reflection_lm=model),
        refiner=RefinerConfig(),
    )

    fitness_fn = create_fitness_function(
        state_tracker=state_tracker,
        timeout=timeout,
        run_logger=run_logger,
    )

    print("\n" + "=" * 70)
    print("Running GEPA Single-Task Optimization with RefinerConfig")
    print("=" * 70 + "\n")

    try:
        result = optimize_anything(
            seed_candidate=seed_candidate,
            evaluator=fitness_fn,
            config=gepa_config,
            objective=f"Optimize circle packing code to maximize sum of circle radii within a unit square for N={NUM_CIRCLES} circles.",
            background=CIRCLE_PACKING_BACKGROUND,
        )

        state_tracker.save_logs()
    except Exception:
        state_tracker.save_logs()
        raise

    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print("Results:", result)


if __name__ == "__main__":
    main()
