# Reliable GPU Kernel Bench

`Reliable GPU Kernel Bench` is a prototype for telemetry-aware GPU kernel benchmarking, finalist-pair rerun allocation, and confidence-gated promotion.

It focuses on one control problem: generated or auto-tuned kernels should not be promoted on the basis of noisy wins caused by thermal drift, contention, clock variation, host jitter, cache effects, or other contaminated benchmark conditions.

Core loop:

- repeated candidate benchmarking
- telemetry capture
- contamination classification
- finalist-pair ambiguity analysis
- rerun allocation only to unresolved finalist pairs
- promote / reject / defer decision

The repo runs on CPU by default for simulation and artifact generation, and is also wired for real Triton tutorial kernels on a GPU machine.

## Start here

CPU simulation:

```powershell
python simulate_results.py --runs 20
python generate_visual_assets.py
```

GPU path:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gpu_flow.ps1
```

Where to look for results:

- single-run outputs: `runs/latest_run.*`
- CPU simulation outputs: `reports/simulation_*`
- GPU batch outputs: `runs/batch_outputs/triton_matmul_generated/`
- visual assets: `reports/visual_assets/`
- docs and summaries: `docs/`

## Why this matters

Most autotuners and generated-kernel pipelines still rank candidates by raw timing aggregates. That is risky when measurements are distorted by unstable runtime conditions. This repo shows a tighter selection model: use telemetry to detect contamination, spend extra benchmark budget only on unresolved finalist pairs, and gate promotion on evidence quality instead of apparent speed alone.

## Quick links

- [Demo packet](docs/demo_packet.html)
- [Key findings](docs/key_findings.html)
- [Technical spec](docs/tech_spec.html)
- [Architecture](docs/architecture.html)
- [Variant search report](docs/variant_search_report.html)
- [Visual gallery](reports/visual_assets/visual_gallery.html)
- [Roadmap](ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## Architecture preview

![Architecture preview](reports/visual_assets/diagrams/control_loop.svg)

## Results snapshot

Latest CPU simulation snapshot:

| Metric | Value |
| --- | ---: |
| Runs | 20 |
| Naive false winners | 12 |
| Correct promotions | 1 |
| False promotions | 0 |
| Defers | 16 |

Selection outcomes:

![Selection outcomes](reports/visual_assets/plots/selection_outcomes_highres.png)

Benchmark story card:

![Benchmark story card](reports/visual_assets/deck/benchmark_story_card.png)

## Repository layout

- `docs/`: human-readable docs, memo-style summaries, and HTML review pages
- `reports/`: generated CPU simulation outputs, plots, and visual assets
- `runs/`: single-run outputs and GPU batch-run outputs
- `examples/`: official Triton tutorial kernels used as benchmarkable workloads
- `tests/`: smoke tests visible from the repo root

## Main scripts

- `engine.py`: core data model and decision logic
- `demo.py`: single-run benchmark flow
- `simulate_results.py`: repeated CPU simulation harness
- `triton_batch_runner.py`: repeated Triton generated-variant batch runner
- `triton_adapter.py`: Triton workload adapter and generated matmul variant definitions
- `setup_gpu.py`: preflight checker for GPU + Triton + tutorial import
- `generate_visual_assets.py`: creates polished diagrams and high-resolution plots
- `variant_search_report.py`: builds the generated-candidate report, optionally enriched with GPU batch results

## Run

CPU demo:

```powershell
python demo.py
```

CPU simulation:

```powershell
python simulate_results.py --runs 20
```

GPU preflight:

```powershell
python setup_gpu.py
```

Generated Triton batch run on a GPU machine:

```powershell
python triton_batch_runner.py --runs 12 --workload matmul --candidate-mode generated
```

One-command GPU flow:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gpu_flow.ps1
```

This runs:

- `setup_gpu.py`
- single Triton matmul
- generated Triton matmul
- Triton generated-variant batch run
- `variant_search_report.py`

## Reports and outputs

Single-run outputs:

- `runs/latest_run.html`
- `runs/latest_run.json`
- `runs/latest_run.txt`

CPU simulation outputs:

- `reports/simulation_results.csv`
- `reports/simulation_summary.json`
- `reports/simulation_summary.html`
- `reports/plots/selection_outcomes.png`
- `reports/plots/final_gain_vs_lcb.png`
- `reports/plots/decision_timeline.png`

GPU batch outputs:

- `runs/batch_outputs/triton_matmul_generated/triton_batch_results.csv`
- `runs/batch_outputs/triton_matmul_generated/triton_batch_summary.json`
- `runs/batch_outputs/triton_matmul_generated/triton_batch_summary.html`
- `runs/batch_outputs/triton_matmul_generated/plots/*.png`

Docs:

- `docs/tech_spec.html`
- `docs/architecture.html`
- `docs/key_findings.html`
- `docs/gpu_run_checklist.html`
- `docs/demo_packet.html`
- `docs/variant_search_report.html`

Visual assets:

```powershell
python generate_visual_assets.py
```

This writes:

- `reports/visual_assets/visual_gallery.html`
- `reports/visual_assets/diagrams/*.svg`
- `reports/visual_assets/plots/*_highres.png`
- `reports/visual_assets/deck/*`

## Tests

Run the smoke tests with:

```powershell
python -m unittest discover -s tests -v
```

## Real Triton workloads wired in

The adapter supports:

- matrix multiplication via `examples/triton_tutorials/03-matrix-multiplication.py`
- fused softmax via `examples/triton_tutorials/02-fused-softmax.py`

The adapter trims each tutorial before its top-level demo and benchmark code, so importing it does not automatically execute tutorial plots or self-tests.

## Small generated-kernel layer

The Triton matmul path also supports:

```powershell
python demo.py --backend triton --workload matmul --candidate-mode generated
```

This mode builds a small fixed-config candidate set from the official Triton matmul kernel search space, including variants with different tile sizes, group sizes, numbers of warps, and pipeline stages. It is the first step toward a Standard Kernel-like automatic candidate generation loop while still using the same contamination-aware promotion engine.

## How to adapt further

You can extend the current setup with:

- additional Triton tutorial kernels launched through Python wrappers
- CUDA kernels invoked through PyTorch extensions, CuPy, or custom bindings
- additional generated variant families for softmax, reduction, or attention kernels

And you can replace the fallback telemetry provider with:

- `nvidia-smi` polling
- NVML bindings
- framework-specific GPU telemetry

The core engine should not need structural changes.

## Sources

- Tutorials index: [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- Matrix multiplication tutorial: [03-matrix-multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- Fused softmax tutorial: [02-fused-softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
