# Cloud Scheduling — CloudCast & Can't Be Late (§5.2)

**Paper claim.** Two cloud-infrastructure algorithms from the ADRS benchmark, both optimized in generalization mode with train/val splits over infrastructure scenarios:

- **CloudCast** (broadcast routing): **40.2% cost savings** over Dijkstra, evolving a shortest-path baseline into a provider-aware Steiner-tree solver that jointly optimizes egress cost and transfer latency. (Figure 3a in the paper.)
- **Can't Be Late** (spot-vs-on-demand scheduling): **7.8% cost savings**, evolving a deadline-check heuristic into an adaptive strategy with state tracking for spot-unavailability patterns, break-even switching cost analysis, and graduated decision thresholds. (Figure 3b in the paper.)

Both top the ADRS leaderboard, outperforming OpenEvolve, ShinkaEvolve, and expert-designed heuristics.

## Code (this folder)

| Subfolder | Contents |
|---|---|
| `cloudcast/` | `main.py` + `evaluator.py` + `cloudcast/` (broadcast simulator with provider profiles, throughput/cost CSVs, multi-cloud config), `requirements.txt` |
| `can_be_late/` | `main.py` + `evaluator.py` + `simulator/` (sky-spot scheduler with real availability traces in `simulator/real_traces.tar.gz`), `trace_dataset.py`, `trace_config.py` |

Each subfolder ships an `optimization_trajectory.png` / `.pdf` corresponding to Figure 3a / 3b in the paper.

## Reproduction

Each subfolder has its own `README.md` with model + dataset setup. Briefly:

```bash
# CloudCast
cd cloudcast
pip install -r requirements.txt
export OPENAI_API_KEY=<your-key>
python main.py

# Can't Be Late
cd can_be_late
python main.py
```

## Logs

The bundled `optimization_trajectory.png` / `.pdf` in each subfolder shows the score-vs-iteration curve for the paper run. The full evolved policy `.py` files and per-scenario benchmark scores live in a separate research repository (`frontiercs`) and are not included here — re-run `main.py` to regenerate the optimized solver.
