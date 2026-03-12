from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List

import psutil
import torch

from engine import (
    CandidateSummary,
    ContaminationClassifier,
    DecisionThresholds,
    NoiseAwareBenchmarkEngine,
    TelemetrySample,
    TrialResult,
)
from triton_adapter import build_triton_candidates, triton_available


torch.set_num_threads(max(1, min(4, psutil.cpu_count(logical=False) or 1)))

M = 128
K = 128
N = 128
A = torch.randn(M, K)
B = torch.randn(K, N)


class DemoTelemetryProvider:
    def __init__(self, noisy_candidates: set[str] | None = None) -> None:
        self.baseline_temp = 64.0
        self.baseline_sm_clock = 1800.0
        self.baseline_mem_clock = 9000.0
        self.noisy_candidates = noisy_candidates or set()

    def sample(self, trial_index: int, candidate_name: str) -> TelemetrySample:
        gpu_metrics = self._query_nvidia_smi()
        cpu = psutil.cpu_percent(interval=None)
        thermal = gpu_metrics["thermal_c"] if gpu_metrics["thermal_c"] is not None else self.baseline_temp + (trial_index * 0.35)
        sm_clock = gpu_metrics["sm_clock_mhz"] if gpu_metrics["sm_clock_mhz"] is not None else self.baseline_sm_clock - (trial_index * 4.0)
        mem_clock = gpu_metrics["mem_clock_mhz"] if gpu_metrics["mem_clock_mhz"] is not None else self.baseline_mem_clock - (trial_index * 6.0)
        active_processes = 1
        synthetic_interference = False

        # Inject contamination into the superficially faster variant so the
        # prototype shows why naive ranking can fail.
        if candidate_name in self.noisy_candidates and trial_index in {0, 1, 2, 4, 8, 9}:
            cpu = 92.0
            thermal += 8.5
            sm_clock -= 140.0
            mem_clock -= 600.0
            active_processes = 2
            synthetic_interference = True

        launch_cpu = psutil.Process().cpu_num() if hasattr(psutil.Process(), "cpu_num") else 0
        return TelemetrySample(
            cpu_load_pct=cpu,
            thermal_c=thermal,
            sm_clock_mhz=sm_clock,
            mem_clock_mhz=mem_clock,
            active_processes=active_processes,
            launch_cpu=launch_cpu,
            numa_node=0,
            synthetic_interference=synthetic_interference,
        )

    def _query_nvidia_smi(self) -> Dict[str, float | None]:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return {"thermal_c": None, "sm_clock_mhz": None, "mem_clock_mhz": None}
        try:
            output = subprocess.check_output(
                [
                    exe,
                    "--query-gpu=temperature.gpu,clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if not output:
                return {"thermal_c": None, "sm_clock_mhz": None, "mem_clock_mhz": None}
            first = output.splitlines()[0]
            temp, sm, mem = [value.strip() for value in first.split(",")]
            return {
                "thermal_c": float(temp),
                "sm_clock_mhz": float(sm),
                "mem_clock_mhz": float(mem),
            }
        except Exception:
            return {"thermal_c": None, "sm_clock_mhz": None, "mem_clock_mhz": None}


def baseline_candidate() -> None:
    torch.mm(A, B)
    time.sleep(0.0010)


def candidate_fast_but_noisy() -> None:
    torch.mm(A, B)
    time.sleep(0.00115)


def candidate_consistent() -> None:
    torch.mm(A, B)
    time.sleep(0.00075)


def candidate_slightly_slow() -> None:
    torch.mm(A, B)
    time.sleep(0.00145)


def run_candidate(fn: Callable[[], None], telemetry: TelemetrySample) -> float:
    start = time.perf_counter()
    fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Synthetic contamination can make a kernel look spuriously faster.
    if telemetry.synthetic_interference:
        elapsed_ms *= 0.58

    return elapsed_ms


def append_trials(
    trials_by_candidate: Dict[str, List[TrialResult]],
    candidates: Dict[str, Callable[[], None]],
    names: List[str],
    start_trial_index: int,
    num_trials: int,
    classifier: ContaminationClassifier,
    telemetry_provider: DemoTelemetryProvider,
) -> None:
    for local_idx in range(num_trials):
        for name in names:
            trial_index = start_trial_index + local_idx
            telemetry = telemetry_provider.sample(trial_index=trial_index, candidate_name=name)
            latency_ms = run_candidate(candidates[name], telemetry)
            label, reasons = classifier.classify(telemetry)
            trials_by_candidate[name].append(
                TrialResult(
                    candidate_name=name,
                    trial_index=trial_index,
                    latency_ms=latency_ms,
                    telemetry=telemetry,
                    label=label,
                    reasons=reasons,
                )
            )


def print_summary(title: str, summaries: Dict[str, CandidateSummary]) -> None:
    print(f"\n{title}")
    for summary in sorted(summaries.values(), key=lambda item: item.name):
        print(
            f"- {summary.name}: "
            f"all_median={summary.median_all_ms:.3f} ms, "
            f"acceptable_median={summary.median_acceptable_ms if summary.median_acceptable_ms is not None else 'NA'} ms, "
            f"acceptable={summary.acceptable_count}, "
            f"contamination_rate={summary.contamination_rate:.2f}"
        )


def build_cpu_candidates() -> Dict[str, Callable[[], None]]:
    return {
        "candidate_baseline": baseline_candidate,
        "candidate_fast_but_noisy": candidate_fast_but_noisy,
        "candidate_consistent": candidate_consistent,
        "candidate_slightly_slow": candidate_slightly_slow,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Noise-aware kernel benchmark prototype")
    parser.add_argument(
        "--backend",
        choices=["cpu", "triton"],
        default="cpu",
        help="Run the CPU demo or a Triton-backed demo on a GPU machine.",
    )
    parser.add_argument(
        "--workload",
        choices=["matmul", "softmax"],
        default="matmul",
        help="Choose the real Triton workload when using --backend triton.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["tutorial", "generated"],
        default="tutorial",
        help="Use the stock tutorial kernels or a small generated fixed-config matmul variant set.",
    )
    return parser.parse_args()


def write_reports(
    args: argparse.Namespace,
    naive: CandidateSummary | None,
    analysis,
    final_analysis,
    decision: str,
    final_summaries: Dict[str, CandidateSummary],
    report_payload: Dict[str, object],
) -> None:
    report_path = Path("kernel_noise_prototype") / "latest_run.txt"
    json_path = Path("kernel_noise_prototype") / "latest_run.json"
    html_path = Path("kernel_noise_prototype") / "latest_run.html"
    lines = [
        "Noise-aware kernel benchmarking demo",
        f"naive_winner={naive.name if naive else 'none'}",
        f"final_decision={decision}",
        "",
    ]
    for summary in final_summaries.values():
        lines.append(
            f"{summary.name}: all_median={summary.median_all_ms:.4f} "
            f"acceptable_median={summary.median_acceptable_ms} "
            f"acceptable={summary.acceptable_count} "
            f"contamination_rate={summary.contamination_rate:.2f}"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    rows = []
    for name, summary in sorted(final_summaries.items()):
        rows.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{summary.median_all_ms:.4f}</td>"
            f"<td>{'NA' if summary.median_acceptable_ms is None else f'{summary.median_acceptable_ms:.4f}'}</td>"
            f"<td>{summary.acceptable_count}</td>"
            f"<td>{summary.contamination_rate:.2f}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Noise-Aware Benchmark Report</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      background: #f6f2ea;
      color: #1f1a17;
      margin: 0;
    }}
    main {{
      max-width: 960px;
      margin: 24px auto;
      background: #fffdf8;
      border: 1px solid #d8cfc4;
      padding: 28px 32px 36px;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.08);
    }}
    h1 {{ margin-top: 0; }}
    .callout {{
      background: #f8f2e9;
      border-left: 4px solid #8c2f1b;
      padding: 12px 14px;
      margin: 16px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
    }}
    th, td {{
      border: 1px solid #d8cfc4;
      padding: 10px 12px;
      text-align: left;
    }}
    th {{
      background: #f4ede3;
    }}
    code {{
      font-family: "Courier New", monospace;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Noise-Aware Benchmark Report</h1>
    <p>This report was generated by the {args.backend}-backed prototype for telemetry-aware contamination filtering, finalist-pair reruns, and promotion gating.</p>
    <div class="callout">
      <strong>Naive raw-median winner:</strong> {report_payload["naive_winner"]}<br>
      <strong>Final decision:</strong> {decision}
    </div>
    <p><strong>Initial finalist analysis:</strong> {analysis.winner_name} vs {analysis.loser_name},
    gain={analysis.rel_gain_pct:.3f}%, lcb={analysis.lower_confidence_bound_pct:.3f}%, unresolved={analysis.unresolved}, reasons={analysis.reasons}</p>
    <p><strong>Final finalist analysis:</strong> {final_analysis.winner_name} vs {final_analysis.loser_name},
    gain={final_analysis.rel_gain_pct:.3f}%, lcb={final_analysis.lower_confidence_bound_pct:.3f}%, unresolved={final_analysis.unresolved}, reasons={final_analysis.reasons}</p>
    <table>
      <thead>
        <tr>
          <th>Candidate</th>
          <th>All-trial median (ms)</th>
          <th>Acceptable-trial median (ms)</th>
          <th>Acceptable trials</th>
          <th>Contamination rate</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")

    print(f"Saved run summary to {report_path}")
    print(f"Saved JSON report to {json_path}")
    print(f"Saved HTML report to {html_path}")


def run_demo(
    backend: str = "cpu",
    workload: str = "matmul",
    candidate_mode: str = "tutorial",
    seed: int = 7,
    write_report_files: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    args = argparse.Namespace(backend=backend, workload=workload, candidate_mode=candidate_mode)
    random.seed(seed)

    if args.backend == "triton":
        if not triton_available():
            raise RuntimeError(
                "Triton backend requested, but Triton is not installed. "
                "See kernel_noise_prototype/MIGRATE_TO_GPU.md."
            )
        candidates = build_triton_candidates(workload=args.workload, candidate_mode=args.candidate_mode)
        noisy_candidates = {name for name in candidates if name.endswith("_noisy")}
    else:
        candidates = build_cpu_candidates()
        noisy_candidates = {"candidate_fast_but_noisy"}

    telemetry_provider = DemoTelemetryProvider(noisy_candidates=noisy_candidates)
    thresholds = DecisionThresholds(
        cpu_load_pct=95.0,
        min_acceptable_trials=10,
        promote_gain_min_pct=2.0,
        promote_lcb_min_pct=1.5,
        contamination_rate_max=0.15,
        finalist_min_acceptable_trials=4,
        base_pair_budget=6,
        contamination_bonus=4,
    )
    classifier = ContaminationClassifier(
        thresholds=thresholds,
        baseline_sm_clock_mhz=telemetry_provider.baseline_sm_clock,
        baseline_mem_clock_mhz=telemetry_provider.baseline_mem_clock,
        baseline_thermal_c=telemetry_provider.baseline_temp,
        allowed_cpu_set=[0, 1, 2, 3],
        required_numa_node=0,
    )
    engine = NoiseAwareBenchmarkEngine(thresholds=thresholds)

    trials_by_candidate: Dict[str, List[TrialResult]] = {name: [] for name in candidates}

    # Warmups
    for fn in candidates.values():
        fn()

    append_trials(
        trials_by_candidate=trials_by_candidate,
        candidates=candidates,
        names=list(candidates.keys()),
        start_trial_index=0,
        num_trials=6,
        classifier=classifier,
        telemetry_provider=telemetry_provider,
    )

    summaries = engine.summarize(trials_by_candidate)
    if verbose:
        print_summary("Initial summaries", summaries)

    naive = engine.naive_winner(summaries)
    if naive and verbose:
        print(f"\nNaive winner by raw median across all trials: {naive.name}")

    finalists = engine.pick_finalists(summaries)
    assert len(finalists) == 2
    analysis = engine.analyze_pair(finalists[0], finalists[1])
    if verbose:
        print(
            f"Initial finalist pair: {analysis.winner_name} vs {analysis.loser_name} | "
            f"gain={analysis.rel_gain_pct:.3f}% | "
            f"lcb={analysis.lower_confidence_bound_pct:.3f}% | "
            f"unresolved={analysis.unresolved} | "
            f"reasons={analysis.reasons}"
        )

    if analysis.unresolved:
        rerun_budget = engine.rerun_budget_for_pair(
            summaries[analysis.winner_name], summaries[analysis.loser_name]
        )
        if verbose:
            print(f"Allocating {rerun_budget} rerun rounds only to finalist pair")
        append_trials(
            trials_by_candidate=trials_by_candidate,
            candidates=candidates,
            names=[analysis.winner_name, analysis.loser_name],
            start_trial_index=6,
            num_trials=rerun_budget,
            classifier=classifier,
            telemetry_provider=telemetry_provider,
        )

    final_summaries = engine.summarize(trials_by_candidate)
    if verbose:
        print_summary("Final summaries", final_summaries)

    final_analysis = engine.analyze_pair(
        final_summaries[analysis.winner_name], final_summaries[analysis.loser_name]
    )
    decision = engine.promotion_decision(final_analysis, final_summaries)

    if verbose:
        print(
            f"\nFinal finalist analysis: {final_analysis.winner_name} vs {final_analysis.loser_name} | "
            f"gain={final_analysis.rel_gain_pct:.3f}% | "
            f"lcb={final_analysis.lower_confidence_bound_pct:.3f}% | "
            f"unresolved={final_analysis.unresolved} | "
            f"reasons={final_analysis.reasons}"
        )
        print(f"Decision: {decision}")

    report_payload = {
        "backend": args.backend,
        "workload": args.workload,
        "candidate_mode": args.candidate_mode,
        "seed": seed,
        "naive_winner": naive.name if naive else None,
        "initial_finalist_analysis": {
            "winner_name": analysis.winner_name,
            "loser_name": analysis.loser_name,
            "rel_gain_pct": analysis.rel_gain_pct,
            "lower_confidence_bound_pct": analysis.lower_confidence_bound_pct,
            "unresolved": analysis.unresolved,
            "reasons": analysis.reasons,
        },
        "final_finalist_analysis": {
            "winner_name": final_analysis.winner_name,
            "loser_name": final_analysis.loser_name,
            "rel_gain_pct": final_analysis.rel_gain_pct,
            "lower_confidence_bound_pct": final_analysis.lower_confidence_bound_pct,
            "unresolved": final_analysis.unresolved,
            "reasons": final_analysis.reasons,
        },
        "decision": decision,
        "summaries": {
            name: {
                "median_all_ms": summary.median_all_ms,
                "median_acceptable_ms": summary.median_acceptable_ms,
                "acceptable_count": summary.acceptable_count,
                "contamination_rate": summary.contamination_rate,
            }
            for name, summary in final_summaries.items()
        },
    }
    if write_report_files:
        write_reports(
            args=args,
            naive=naive,
            analysis=analysis,
            final_analysis=final_analysis,
            decision=decision,
            final_summaries=final_summaries,
            report_payload=report_payload,
        )
    return report_payload


def main() -> None:
    args = parse_args()
    run_demo(
        backend=args.backend,
        workload=args.workload,
        candidate_mode=args.candidate_mode,
        seed=7,
        write_report_files=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
