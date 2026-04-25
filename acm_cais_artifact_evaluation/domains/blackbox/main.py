#!/usr/bin/env python3
"""Blackbox optimization with GEPA + external budget tracking.

Budget is tracked externally via crash-proof call logs. Even if generated
code fails, the objective calls it made are counted and deducted from the
total budget. Each candidate gets the remaining budget.
"""

import json
import time
from pathlib import Path

from config import parse_arguments, get_log_directory
from utils import (
    BudgetTracker,
    execute_code,
    extract_best_xs,
    serialize_attempts,
    SEED_CODE,
    OBJECTIVE,
    BACKGROUND,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)


def main():
    args = parse_arguments()
    log_dir = get_log_directory(args)
    budget = BudgetTracker(args.evaluation_budget, args.num_proposals)

    print(f"GEPA — Problem {args.problem_index}")
    print(f"Model: {args.llm_model}")
    print(f"Proposals: {args.num_proposals}, Total budget: {args.evaluation_budget}")
    print(f"Timeout: {args.timeout_per_trial}s/trial")
    print(f"Output: {log_dir}")

    # Persistent per-call eval log
    eval_log_path = Path(log_dir) / "eval_log.jsonl"
    eval_count = 0
    best_score = float("inf")
    t_start = time.time()

    def flush_eval_log(logged_attempts, candidate_num):
        """Append per-call entries from one candidate to the persistent log."""
        nonlocal eval_count, best_score
        with open(eval_log_path, "a") as f:
            for entry in logged_attempts:
                eval_count += 1
                score = entry["score"]
                if score < best_score:
                    best_score = score
                f.write(json.dumps({
                    "eval_num": eval_count,
                    "time": entry.get("time"),
                    "elapsed": entry["time"] - t_start if entry.get("time") else None,
                    "score": score,
                    "best_score": best_score,
                    "candidate_num": candidate_num,
                    "budget_used": budget.used,
                    "budget_total": budget.total,
                }) + "\n")

    def evaluator(candidate, opt_state):
        if budget.remaining <= 0:
            return (-1e9, {"score": -1e9, "Error": "No budget remaining"})

        candidate_budget = budget.per_candidate
        timeout = candidate_budget * args.timeout_per_trial
        best_xs = extract_best_xs(opt_state, top_k=args.top_k_trajectory)

        result = execute_code(
            code=candidate["code"],
            problem_index=args.problem_index,
            timeout=timeout,
            budget=candidate_budget,
            best_xs=best_xs,
        )

        budget.record(result)
        logged = result["logged_attempts"]
        sorted_logged = sorted(logged, key=lambda a: a["score"]) if logged else []

        flush_eval_log(logged, candidate_num=budget.candidates)

        side_info = {
            "score": result["score"],
            "all_trials": serialize_attempts(sorted_logged),
            "Stdout": result.get("stdout", ""),
            "Error": result.get("error", ""),
            "Traceback": result.get("traceback", ""),
            "budget_total": budget.total,
            "budget_used": budget.used,
            "proposal_total": args.num_proposals,
            "proposal_completed": budget.candidates,
        }
        return (result["score"], side_info)

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            seed=args.seed,
            max_candidate_proposals=args.num_proposals,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=args.llm_model,
        ),
        refiner=None,
    )

    t0 = time.time()
    optimize_anything(
        seed_candidate={"code": SEED_CODE},
        evaluator=evaluator,
        config=gepa_config,
        objective=OBJECTIVE,
        background=BACKGROUND,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"FINAL: {budget.used}/{budget.total} evals, {budget.candidates} candidates, {elapsed:.1f}s")

    Path(log_dir, "results.json").write_text(json.dumps({
        "problem_index": args.problem_index,
        "total_evals": budget.used,
        "elapsed": elapsed,
    }, indent=2))

    print(f"Results saved to {log_dir}/results.json")


if __name__ == "__main__":
    main()
