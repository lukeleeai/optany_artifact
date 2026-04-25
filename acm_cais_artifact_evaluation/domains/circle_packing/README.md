# Circle Packing (§5.6)

**Paper claim.** Pack n=26 unit circles in the unit square to maximize the sum of radii. optany reaches **2.63598+**, beating AlphaEvolve (2.6358), OpenEvolve, and ShinkaEvolve (2.635978). The optimized algorithm is a bilevel optimizer: an LP over radii with dual-variable gradients for L-BFGS-B center optimization, augmented by CMA-ES exploration and diverse seeding.

## Code (this folder)

- `main.py`, `utils.py`, `requirements.txt` — runnable optany pipeline (copied from `examples/circle_packing/`).

## Reproduction

`main.py` is wired to use `./logs/` as the GEPA run directory. Because the bundled `logs/gepa_state.bin` already exists, GEPA will load that checkpoint and resume from the paper run rather than starting from scratch — letting reviewers verify the optimized solver immediately.

```bash
export OPENAI_API_KEY=<your-key>
uv run --with-requirements requirements.txt python main.py    # resumes from bundled state
```

To run a fresh optimization instead, delete (or move) `logs/gepa_state.bin` first.

## What's in `logs/`

| File | Purpose |
|---|---|
| `state_tracker_logs.json` | Per-iteration optimizer state. Best recorded score: `2.635983362593453` (search the file for this string). Contains the evolved Python solver code at each iteration along with its evaluated sum-of-radii |
| `generated_best_outputs_valset/` | Per-task outputs from the best candidate |

`gepa_state.bin` is the full GEPA optimizer checkpoint (3.2 MB) — restore with `gepa.GEPAState.load()` to inspect the candidate pool. `state_tracker_logs.json` is human-readable and contains the full score trajectory and best artifact code.

## Comparison table (from `state_tracker_logs.json`)

| Rank | Method | Sum of Radii |
|---|---|---|
| 1 | optany (this run) | **2.635983362…** |
| 2 | ShinkaEvolve (500 iters) | 2.635977709… |
| 3 | AlphaEvolve (published) | 2.6358 |
