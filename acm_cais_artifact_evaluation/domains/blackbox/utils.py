"""Utilities for polynomial optimization: code execution and seed templates."""

import numpy as np
import json
import os
import tempfile
import time

from problems import problems

from gepa.utils.code_execution import execute_code as _execute_code, ExecutionMode


class BudgetTracker:
    """Counts actual objective calls even on crash."""

    def __init__(self, total, num_candidates):
        self.total = total
        self.num_candidates = num_candidates
        self.used = 0
        self.candidates = 0

    @property
    def remaining(self):
        return self.total - self.used

    @property
    def per_candidate(self):
        left = self.num_candidates - self.candidates
        if left <= 0:
            return 0
        return self.remaining // left

    def record(self, result):
        self.candidates += 1
        n = result["actual_call_count"]
        self.used += n
        return n



def execute_code(
    code: str,
    problem_index: int,
    timeout: int,
    budget: int | None = None,
    best_xs: list[dict] | None = None,
    seed: int = 0,
    extra_config: dict | None = None,
) -> dict:
    """Execute optimization code and return structured result.

    Wraps objective_function with a call logger that writes to a temp file.
    Even if the code crashes, we know how many evals were used.

    Returns dict with:
        - success, score, all_attempts, serialized_attempts (as before)
        - logged_attempts: list of {x, score} from the call log (crash-proof)
        - actual_call_count: number of objective calls actually made
    """
    fn = problems[problem_index]

    # Crash-proof call log: each objective call writes a line to this file.
    # Survives subprocess crashes since writes are flushed per call.
    call_log_fd, call_log_path = tempfile.mkstemp(suffix=".jsonl")
    os.close(call_log_fd)

    def objective_function(x):
        score = fn.do_evaluate(np.array(x))
        with open(call_log_path, "a") as f:
            f.write(json.dumps({
                "x": x.tolist() if hasattr(x, "tolist") else list(x),
                "score": float(score),
                "time": time.time(),
            }) + "\n")
        return score

    config = {"bounds": fn.bounds, "dim": fn.dim}
    if budget is not None:
        config["budget"] = budget
    if extra_config:
        config.update(extra_config)

    result = _execute_code(
        code=code,
        timeout=timeout,
        mode=ExecutionMode.SUBPROCESS,
        entry_point="solve",
        entry_point_kwargs={
            "objective_function": objective_function,
            "config": config,
            "best_xs": best_xs or [],
        },
        seed=seed,
    )

    # Read call log (crash-proof — has entries even if code failed)
    logged_attempts = []
    try:
        with open(call_log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    logged_attempts.append(json.loads(line))
    except Exception:
        pass
    finally:
        os.unlink(call_log_path)

    base = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "logged_attempts": logged_attempts,
        "actual_call_count": len(logged_attempts),
    }

    fail = {
        "success": False,
        "score": -1e9,
        "all_attempts": [],
        "serialized_attempts": [],
        **base,
    }

    if not result.success:
        return {**fail, "error": result.error or "Execution failed", "traceback": result.traceback or ""}

    ret = result.variables.get("__return__")
    if not isinstance(ret, dict) or "x" not in ret or "score" not in ret or "all_attempts" not in ret:
        return {**fail, "error": "solve() must return {'x': array, 'score': float, 'all_attempts': [...]}"}

    all_attempts = ret["all_attempts"]
    return {
        "success": True,
        "score": -ret["score"],
        "all_attempts": all_attempts,
        "serialized_attempts": serialize_attempts(all_attempts),
        **base,
    }



def extract_best_xs(opt_state, top_k: int = 1000) -> list[dict]:
    """Extract best_xs from best evaluations, sorted by score (best first)."""
    if opt_state is None:
        return []
    best_example_evals = getattr(opt_state, "best_example_evals", opt_state)
    all_attempts = []
    for e in best_example_evals or []:
        side_info = e.get("side_info", {})
        all_attempts.extend(side_info.get("all_trials", []))
    sorted_attempts = sorted(all_attempts, key=lambda t: t["score"])[:top_k]
    return [{"x": np.array(t["x"]), "score": t["score"]} for t in sorted_attempts]


def serialize_attempts(attempts):
    return [
        {
            "x": a["x"].tolist() if hasattr(a["x"], "tolist") else a["x"],
            "score": a["score"],
        }
        for a in attempts
    ]




# =============================================================================
# SEED CODE
# =============================================================================

SEED_CODE = '''
import numpy as np

def solve(objective_function, config, best_xs=None):
    bounds = np.array(config['bounds'])
    all_attempts = []

    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
    score = objective_function(x)
    all_attempts.append({"x": x.copy(), "score": score})

    return {"x": x, "score": score, "all_attempts": all_attempts}
'''


# =============================================================================
# PROMPTS
# =============================================================================

OBJECTIVE = "Evolve Python code that minimizes a blackbox objective function using the available evaluation budget efficiently."

BACKGROUND = """
You are optimizing code that solves blackbox minimization problems (lower is better).

## Function Signature
```python
def solve(objective_function, config, best_xs=None):
    # config contains: bounds (array of [min, max] per dim), dim (int), budget (int)
    # best_xs: list of {"x": array, "score": float} sorted by score (best first)
    # Returns: {"x": best_x, "score": best_score, "all_attempts": [{"x": x, "score": score}, ...]}
```

## Code Requirements
- Each attempt in all_attempts must have "x" (numpy array) and "score" (float)
- Use `objective_function(x)` to evaluate candidates (lower score is better)
- Stay within `config['budget']` calls
- Full use of all the allowed evaluation budget leads to better performance
- Use `best_xs` to leverage previous evaluation data (if available)

## Trajectory Data
`best_xs` provides all the previous (x, score) evaluations sorted by score — this data is free (zero budget cost).
Use it to build surrogates, find diverse starting points, and avoid already-explored regions.

## Strategy Ideas
- Surrogate-guided search (GP/RBF fitted on best_xs, optimize acquisition function)
- Multi-start local search from diverse seeds (not just the best point)
- Population-based methods (CMA-ES, differential evolution)
- Hybrid: global exploration + local refinement with budget splitting

## Available Libraries
Any package is ready to use.

## Output Format
Provide the improved code in a single code block with triple backticks.
"""
