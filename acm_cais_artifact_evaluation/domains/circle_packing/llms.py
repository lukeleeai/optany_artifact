"""LLM configurations for circle packing single-task optimization."""

CIRCLE_PACKING_BACKGROUND = """
Make BREAKTHROUGH improvements by trying fundamentally different approaches.

Pack 26 non-overlapping circles inside a UNIT SQUARE [0,1] x [0,1].

SCORING: Sum of all circle radii (higher is better!)

CRITICAL CODE FORMAT:
- Function name MUST be: `def main(timeout, current_best_solution):`
- `current_best_solution` is a numpy array of shape (26, 3) or None.
- Return a dictionary with:
    - 'circles': numpy array shape (26, 3) where each row is (x, y, radius)
    - 'all_scores': list of floats (even if just one score)

CRITICAL CONSTRAINTS:
1. All circles fully inside [0,1]×[0,1]: 0 ≤ x-r, x+r ≤ 1 and 0 ≤ y-r, y+r ≤ 1
2. No overlaps: distance between centers ≥ sum of radii
3. Your code should run in <550 seconds. Otherwise, the score will be 0.

INNOVATION STRATEGIES:
1. **Algorithmic diversity**: Physics-based, optimization-based, geometric, hybrid, meta-heuristics
2. **Geometric insights**: Hexagonal patterns, corner utilization, variable radii
3. **Optimization techniques**: Multiple restarts, hierarchical approaches, gradient-free methods
4. **Hyperparameter auto-tuning**: Use optuna/hyperopt to find best parameters automatically
5. Imagine you have all the packages available (optuna, scipy, etc.) in the environment already and freely explore any of the packages you need.

ANALYSIS STRATEGY:
1. If scores plateau → try fundamentally different algorithm
2. If errors persist → address root cause, don't just patch
3. The refiner LLM will handle the refinement process, so you focus on making a big leap in the global strategy.

OUTPUT REQUIREMENTS:
- Return ONLY executable Python code (no markdown, no explanations)
- Focus on BREAKTHROUGH ideas, not incremental tweaks
"""
