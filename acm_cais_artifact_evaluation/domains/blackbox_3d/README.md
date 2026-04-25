# Black-box Math & 3D Modeling (Appendices B & C)

## Appendix B — Black-box Mathematical Optimization

**Paper claim.** Demonstration that optany matches or outperforms Optuna on numerical optimization tasks by treating the candidate solution code as the artifact and optimizer feedback as Side Information.

### Code (this folder)

- `main.py`, `utils.py` — runnable optany pipeline (copied from `examples/blackbox/`)
- `evalset/` — evaluation problems

### Reproduction

```bash
export OPENAI_API_KEY=<your-key>
uv run python main.py
```

## Appendix C — 3D Modeling (Seedless Generation)

**Paper claim.** In seedless mode (no starting artifact), optany bootstraps a working `build123d` + `pyrender` pipeline for generating a 3D unicorn from only a natural-language `objective` and a `background` description of the available libraries.

### Code (this folder)

`3d_unicorn_optimization.ipynb` — runnable notebook (copied from `docs/docs/tutorials/`).

### Reproduction

Open the notebook in Jupyter and execute. Requires `build123d`, `pyrender`, `trimesh`, `numpy`, and a Vision-capable LLM (e.g. Claude Opus 4.6 as proposer, plus a VLM evaluator).

## Logs

Not bundled in this artifact. The 3D unicorn appendix figure in the paper is reproducible from the notebook's output cells.
