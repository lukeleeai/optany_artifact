# 3D Modeling — Seedless Generation (Appendix C)

**Paper claim.** In seedless mode (no starting artifact), `optimize_anything` bootstraps a working `build123d` + `pyrender` pipeline for generating a 3D unicorn from only a natural-language `objective` and a `background` description of the available libraries.

## Code (this folder)

- `3d_unicorn_optimization.ipynb` — runnable notebook (copied from `docs/docs/tutorials/`).

## Reproduction

Open the notebook in Jupyter and execute. Requires `build123d`, `pyrender`, `trimesh`, `numpy`, and a Vision-capable LLM (e.g. Claude Opus 4.6 as proposer, plus a VLM evaluator).

The 3D unicorn appendix figure in the paper is reproducible from the notebook's output cells.
