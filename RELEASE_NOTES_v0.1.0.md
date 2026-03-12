# v0.1.0

Initial public release of `Reliable GPU Kernel Bench`.

## Highlights

- contamination-aware kernel benchmarking engine
- telemetry-derived contamination classification
- finalist-pair ambiguity analysis with selective reruns
- confidence-gated `promote / defer / reject` decision logic
- CPU-backed simulation harness with CSV, JSON, HTML, and PNG outputs
- GPU-ready Triton integration for official matmul and fused-softmax tutorial kernels
- small generated matmul candidate layer based on fixed Triton kernel variants
- technical specification, architecture diagrams, findings page, and GPU run checklist

## Included artifacts

- `demo_packet.html`
- `key_findings.html`
- `tech_spec.html`
- `architecture.html`
- `gpu_run_checklist.html`
- `simulation_summary.html`
- `plots/*.png`

## Notes

This release is prototype-grade. The CPU simulation demonstrates the selection
logic and contamination-aware control loop. Real Triton execution requires a
GPU machine with Triton and CUDA-capable PyTorch installed.
