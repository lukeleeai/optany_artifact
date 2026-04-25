# ACM CAIS '26 Artifact Evaluation — `optany`

Artifact for the paper **"optany: Unified Text Optimization can Outperform Specialized Systems"** (ACM CAIS 2026).

## What this artifact contains

- `paper.pdf` — the submission PDF
- `domains/` — one self-contained subdirectory per experimental domain claimed in the paper. Each domain folder includes:
  - `README.md` (paper claim, reproduction command, what's in `logs/` if bundled)
  - The runnable code (e.g. `main.py`, `utils.py`)
  - `logs/` with the actual run outputs for `arc_agi/`, `aime_math/`, `circle_packing/`, `kernelbench/` — the four domains where the bundled `gepa_state.bin` matches the paper's headline numbers and `python main.py` auto-resumes from it.

## Quick repository map

| Paper section | Folder | Logs included? |
|---|---|---|
| §5.1 Coding Agent Skills | `domains/gskill/` | No |
| §5.2 Cloud Scheduling (CloudCast, Can't Be Late) | `domains/cloud_scheduling/` | Trajectory plots only |
| §5.3 ARC-AGI Agent Architecture | `domains/arc_agi/` | **Yes** |
| §5.4 AIME Prompt Optimization | `domains/aime_math/` | **Yes** |
| §5.5 CUDA Kernel Generation (KernelBench) | `domains/kernelbench/` | **Yes** (V100 required to re-run) |
| §5.6 Circle Packing (n=26) | `domains/circle_packing/` | **Yes** |
| §5.7 SVG / CAD Image Generation | `domains/svg_cad/` | No |
| Appendix B Black-box Math Optimization | `domains/blackbox_3d/` | No |
| Appendix C 3D Modeling (seedless) | `domains/blackbox_3d/` (notebook) | No |

§5.5 (CUDA Kernel Generation, KernelBench) requires a V100 32GB GPU to re-run; logs and code are bundled, see `domains/kernelbench/README.md`.

## Setup

```bash
uv sync --extra dev
```

API keys (per the domain READMEs) and any domain-specific dependencies are listed inside each `domains/<name>/README.md`.

## Badges requested

- **Artifact Available, Functional, and Results Reproduced** — for ARC-AGI (§5.3), AIME (§5.4), Circle Packing (§5.6). Each ships with bundled GEPA state matching the paper's headline numbers, and `python main.py` auto-resumes from that state.
- **Artifact Available and Functional** — for KernelBench (§5.5) (logs and state bundled, but re-running needs a V100 GPU), Coding Agent Skills (§5.1), Cloud Scheduling (§5.2), SVG/CAD (§5.7), Black-box / 3D (Apps. B/C).

## Reproduction summary

For the three "Results Reproduced" domains, the workflow is:

```bash
uv sync --extra dev
cd domains/<name>
export <PROVIDER>_API_KEY=<your-key>     # see the domain README
uv run python main.py
```

`main.py` is wired so `EngineConfig.run_dir` points at `./logs/`. Because the bundled `logs/gepa_state.bin` exists, GEPA loads it and resumes the paper run rather than starting from scratch — the optimized artifact is verifiable immediately.

To run a fresh optimization instead, move or delete `logs/gepa_state.bin` first.
