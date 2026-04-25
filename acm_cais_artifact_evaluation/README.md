# ACM CAIS '26 Artifact Evaluation — `optany`

Artifact for the paper **"optany: Unified Text Optimization can Outperform Specialized Systems"** (ACM CAIS 2026).

## What this artifact contains

- `domains/` — one self-contained subdirectory per experimental domain claimed in the paper. Each domain folder is independent and includes its own `README.md`, runnable code, and any artifacts needed to verify the paper's claims.

## Quick repository map

| Paper section | Folder |
|---|---|
| §5.1 Coding Agent Skills | `domains/gskill/` |
| §5.2 Cloud Scheduling (CloudCast, Can't Be Late) | `domains/cloud_scheduling/` |
| §5.3 ARC-AGI Agent Architecture | `domains/arc_agi/` |
| §5.4 AIME Prompt Optimization | `domains/aime_math/` |
| §5.5 CUDA Kernel Generation (KernelBench) | `domains/kernelbench/` |
| §5.6 Circle Packing (n=26) | `domains/circle_packing/` |
| §5.7 SVG / CAD Image Generation | `domains/svg_cad/` |
| Appendix B Black-box Math Optimization | `domains/blackbox/` |
| Appendix C 3D Modeling (seedless) | `domains/modeling_3d/` |

## Setup

```bash
uv sync --extra dev
```

API keys and any domain-specific dependencies are listed inside each `domains/<name>/README.md`.

## Reproduction

For each domain, the workflow is:

```bash
uv sync --extra dev
cd domains/<name>
export <PROVIDER>_API_KEY=<your-key>     # see the domain README
uv run python main.py
```

Where bundled state is provided, `main.py` is wired so `EngineConfig.run_dir` points at `./logs/`, and GEPA resumes from that checkpoint rather than starting from scratch — the optimized artifact is verifiable immediately. To run a fresh optimization, move or delete `logs/gepa_state.bin` first.

§5.5 (CUDA Kernel Generation, KernelBench) requires a V100 32GB GPU to re-run; see `domains/kernelbench/README.md`.

## Badges requested

Artifact Available, Functional, and Results Reproduced.
