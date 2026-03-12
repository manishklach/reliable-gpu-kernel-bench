from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from demo import run_demo


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated Triton-backed benchmark sessions for generated matmul candidates."
    )
    parser.add_argument("--runs", type=int, default=12, help="Number of repeated Triton sessions.")
    parser.add_argument(
        "--workload",
        choices=["matmul"],
        default="matmul",
        help="Currently only matmul generated variants are supported for batch Triton runs.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["generated", "tutorial"],
        default="generated",
        help="Generated is recommended for Triton batch runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="batch_outputs/triton_matmul_generated",
        help="Output directory relative to the repo root.",
    )
    return parser.parse_args()


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    decisions = [str(record["decision"]) for record in records]
    naive_winners = [str(record["naive_winner"]) for record in records if record["naive_winner"]]
    promote_records = [d for d in decisions if d.startswith("promote:")]
    defer_records = [d for d in decisions if d.startswith("defer:")]
    reject_records = [d for d in decisions if d.startswith("reject:")]

    final_lcbs = [
        float(record["final_finalist_analysis"]["lower_confidence_bound_pct"])  # type: ignore[index]
        for record in records
    ]
    final_gains = [
        float(record["final_finalist_analysis"]["rel_gain_pct"])  # type: ignore[index]
        for record in records
    ]
    winner_contamination = []
    for record in records:
        winner = str(record["final_finalist_analysis"]["winner_name"])  # type: ignore[index]
        summaries = record["summaries"]  # type: ignore[assignment]
        winner_contamination.append(float(summaries[winner]["contamination_rate"]))  # type: ignore[index]

    return {
        "runs": len(records),
        "backend": "triton",
        "workload": records[0]["workload"] if records else "matmul",
        "candidate_mode": records[0]["candidate_mode"] if records else "generated",
        "promote_count": len(promote_records),
        "defer_count": len(defer_records),
        "reject_count": len(reject_records),
        "naive_winner_counts": dict(Counter(naive_winners)),
        "decision_counts": dict(Counter(decisions)),
        "avg_final_gain_pct": round(mean(final_gains), 4) if final_gains else 0.0,
        "avg_final_lcb_pct": round(mean(final_lcbs), 4) if final_lcbs else 0.0,
        "avg_final_winner_contamination_rate": round(mean(winner_contamination), 4)
        if winner_contamination
        else 0.0,
    }


def create_plots(records: list[dict[str, object]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    naive_counts = Counter(str(record["naive_winner"]) for record in records if record["naive_winner"])
    decision_counts = Counter(str(record["decision"]) for record in records)
    gains = [float(record["final_finalist_analysis"]["rel_gain_pct"]) for record in records]  # type: ignore[index]
    lcbs = [
        float(record["final_finalist_analysis"]["lower_confidence_bound_pct"])  # type: ignore[index]
        for record in records
    ]
    colors = [
        "#2a9d8f" if str(record["decision"]).startswith("promote:") else "#bc6c25"
        for record in records
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(list(naive_counts.keys()), list(naive_counts.values()), color="#577590")
    axes[0].set_title("Naive Winner Frequency")
    axes[0].set_ylabel("Runs")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(list(decision_counts.keys()), list(decision_counts.values()), color="#6b8e23")
    axes[1].set_title("Final Decision Frequency")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plots_dir / "selection_outcomes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(gains, lcbs, c=colors, alpha=0.85, edgecolors="black", linewidths=0.3)
    ax.axvline(2.0, color="#444444", linestyle="--", linewidth=1.0, label="Gain gate")
    ax.axhline(1.5, color="#8c2f39", linestyle="--", linewidth=1.0, label="LCB gate")
    ax.set_title("Final Pair Gain vs Lower Confidence Bound")
    ax.set_xlabel("Measured gain (%)")
    ax.set_ylabel("Lower confidence bound (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "final_gain_vs_lcb.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    seeds = [int(record["seed"]) for record in records]
    statuses = [2 if str(record["decision"]).startswith("promote:") else 1 for record in records]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.scatter(seeds, statuses, c=colors, s=70, edgecolors="black", linewidths=0.3)
    ax.set_title("Per-Run Final Outcome Timeline")
    ax.set_xlabel("Seed / Run")
    ax.set_yticks([1, 2], labels=["Defer/Reject", "Promote"])
    ax.set_ylim(0.5, 2.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "decision_timeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_html(summary: dict[str, object], records: list[dict[str, object]], output_dir: Path) -> str:
    summary_rows = [f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in summary.items()]
    record_rows = []
    for record in records:
        record_rows.append(
            "<tr>"
            f"<td>{record['seed']}</td>"
            f"<td>{record['naive_winner']}</td>"
            f"<td>{record['decision']}</td>"
            f"<td>{record['final_finalist_analysis']['winner_name']}</td>"
            f"<td>{record['final_finalist_analysis']['loser_name']}</td>"
            f"<td>{record['final_finalist_analysis']['rel_gain_pct']:.3f}</td>"
            f"<td>{record['final_finalist_analysis']['lower_confidence_bound_pct']:.3f}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Triton Batch Benchmark Summary</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      background: #f6f2ea;
      color: #1f1a17;
      margin: 0;
    }}
    main {{
      max-width: 1080px;
      margin: 24px auto;
      background: #fffdf8;
      border: 1px solid #d8cfc4;
      padding: 28px 32px 36px;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
    }}
    th, td {{
      border: 1px solid #d8cfc4;
      padding: 8px 10px;
      text-align: left;
    }}
    th {{
      background: #f4ede3;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #d8cfc4;
      margin-top: 14px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Triton Batch Benchmark Summary</h1>
    <p>Generated with repeated Triton-backed runs for {summary['candidate_mode']} {summary['workload']} candidates.</p>
    <p><strong>Output directory:</strong> {output_dir}</p>
    <img src="plots/selection_outcomes.png" alt="Selection outcomes">
    <img src="plots/final_gain_vs_lcb.png" alt="Final gain vs LCB">
    <img src="plots/decision_timeline.png" alt="Decision timeline">
    <table>
      <tbody>
        {''.join(summary_rows)}
      </tbody>
    </table>
    <h2>Per-Run Results</h2>
    <table>
      <thead>
        <tr>
          <th>Seed</th>
          <th>Naive Winner</th>
          <th>Decision</th>
          <th>Final Winner</th>
          <th>Final Loser</th>
          <th>Gain %</th>
          <th>LCB %</th>
        </tr>
      </thead>
      <tbody>
        {''.join(record_rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_dir = BASE_DIR / args.output_dir
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for idx in range(args.runs):
        payload = run_demo(
            backend="triton",
            workload=args.workload,
            candidate_mode=args.candidate_mode,
            seed=100 + idx,
            write_report_files=False,
            verbose=False,
        )
        records.append(payload)

    summary = summarize(records)
    create_plots(records, plots_dir)

    csv_path = output_dir / "triton_batch_results.csv"
    json_path = output_dir / "triton_batch_summary.json"
    html_path = output_dir / "triton_batch_summary.html"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "naive_winner",
                "decision",
                "final_winner",
                "final_loser",
                "final_gain_pct",
                "final_lcb_pct",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["seed"],
                    record["naive_winner"],
                    record["decision"],
                    record["final_finalist_analysis"]["winner_name"],  # type: ignore[index]
                    record["final_finalist_analysis"]["loser_name"],  # type: ignore[index]
                    record["final_finalist_analysis"]["rel_gain_pct"],  # type: ignore[index]
                    record["final_finalist_analysis"]["lower_confidence_bound_pct"],  # type: ignore[index]
                ]
            )

    json_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    html_path.write_text(render_html(summary, records, output_dir), encoding="utf-8")

    print("Triton batch run complete")
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved JSON summary to {json_path}")
    print(f"Saved HTML summary to {html_path}")


if __name__ == "__main__":
    main()
