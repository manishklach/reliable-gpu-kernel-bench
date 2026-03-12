# Roadmap

## Near term

- run the Triton flow on a real GPU machine
- collect first real `runs/batch_outputs/triton_matmul_generated` result bundle
- enrich `docs/variant_search_report.html` with real GPU batch outcomes
- tune contamination thresholds using real telemetry rather than CPU-only fallback behavior

## Next

- add more generated kernel families beyond matmul
- add richer telemetry collection through NVML or vendor-specific APIs
- compare generated variants against stronger CUDA baselines
- add batch-run summaries across multiple GPUs or interference conditions

## Longer term

- add automated kernel search over larger Triton or CUDA parameter spaces
- support correctness checks and promotion gating in the same loop
- integrate multi-objective selection using latency, variance, and power
