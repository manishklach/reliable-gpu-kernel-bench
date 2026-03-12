$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

Write-Host "Reliable GPU Kernel Bench - GPU flow" -ForegroundColor Cyan
Write-Host "Repo root: $repoRoot"
Write-Host ""

Write-Host "[1/5] GPU preflight" -ForegroundColor Yellow
python .\setup_gpu.py

Write-Host ""
Write-Host "[2/5] Single Triton matmul run" -ForegroundColor Yellow
python .\demo.py --backend triton --workload matmul

Write-Host ""
Write-Host "[3/5] Generated Triton matmul run" -ForegroundColor Yellow
python .\demo.py --backend triton --workload matmul --candidate-mode generated

Write-Host ""
Write-Host "[4/5] Triton generated-variant batch run" -ForegroundColor Yellow
python .\triton_batch_runner.py --runs 12 --workload matmul --candidate-mode generated

Write-Host ""
Write-Host "[5/5] Variant search report refresh" -ForegroundColor Yellow
python .\variant_search_report.py

Write-Host ""
Write-Host "GPU flow complete." -ForegroundColor Green
Write-Host "Open these files next:"
Write-Host " - latest_run.html"
Write-Host " - batch_outputs\triton_matmul_generated\triton_batch_summary.html"
Write-Host " - variant_search_report.html"
