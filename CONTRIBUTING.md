# Contributing

## Getting started

1. Run the CPU simulation first:

```powershell
python simulate_results.py --runs 20
python generate_visual_assets.py
```

2. Run the smoke tests:

```powershell
python -m unittest discover -s tests -v
```

3. If you have a GPU machine, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_gpu_flow.ps1
```

## Repository conventions

- keep runnable code at the repo root
- put generated reports in `reports/`
- put execution outputs in `runs/`
- put narrative and review docs in `docs/`
- put example kernels in `examples/`
- keep tests visible under `tests/`

## When adding new kernel variants

- define the variant or generator in `triton_adapter.py`
- make sure it can flow through the existing benchmark engine
- update `variant_search_report.py` if the candidate set changes materially
- regenerate reports and visuals if public-facing outputs change
