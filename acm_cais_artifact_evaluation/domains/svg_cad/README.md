# SVG / CAD Image Generation (§5.7)

**Paper claim.** Multi-task search over visual aspects produces images that humans unanimously prefer over zero-shot baselines, across four goals (Table 3): pelican on a bicycle (SVG, 12 aspects), a high-quality 3D unicorn (CAD via build123d, 4 aspects), an octopus on a grand pipe organ (SVG, 13 aspects), and a sloth steering an excavator at golden hour (SVG, 13 aspects).

## Code (this folder)

- `svg_cad_tutorial.md` — the full optany blog post containing runnable code for the SVG (pelican on a bicycle) evaluator, VLM scoring, and multi-task `optimize_anything` call. (Copied from `docs/docs/blog/posts/2026-02-18-introducing-optimize-anything/index.md`.)
- `3d_unicorn_optimization.ipynb` — runnable notebook for the 3D unicorn CAD case (copied from `docs/docs/tutorials/`).
- `images/` — optimized + zero-shot SVG outputs from the paper (`optimized_pelican_gemini_3_flash.svg`, `gemini-3-pro-best-pelican.svg`, `gemini_3_flash_zero_shot.svg`, etc.).

## Reproduction

The pelican code requires a Vision-capable LLM (e.g. `vertex_ai/gemini-3-flash-preview`) for both evaluator scoring and reflection. The 3D unicorn additionally needs `build123d`, `pyrender`, and `trimesh` installed.

## Logs

Not bundled in this artifact (optimization runs produce large multi-modal trajectories; the optimized SVG/CAD outputs in `images/` are the visual artifacts of those runs).
