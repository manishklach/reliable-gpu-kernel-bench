from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from demo import run_demo


DEFAULT_TRUE_WINNER = "candidate_consistent"
BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated simulated benchmark sessions and compare naive vs contamination-aware selection."
    )
    parser.add_argument("--runs", type=int, default=25, help="Number of simulated sessions to run.")
    parser.add_argument(
        "--true-winner",
        default=DEFAULT_TRUE_WINNER,
        help="Expected true winner for the CPU simulation.",
    )
    return parser.parse_args()


def summarize(records: list[dict[str, object]], true_winner: str) -> dict[str, object]:
    promote_records = [r for r in records if str(r["decision"]).startswith("promote:")]
    defer_records = [r for r in records if str(r["decision"]).startswith("defer:")]
    reject_records = [r for r in records if str(r["decision"]).startswith("reject:")]

    naive_matches_true = sum(1 for r in records if r["naive_winner"] == true_winner)
    final_promotes_true = sum(
        1 for r in promote_records if str(r["decision"]).split(":", 1)[1] == true_winner
    )
    naive_false_winners = sum(1 for r in records if r["naive_winner"] != true_winner)
    final_false_promotions = sum(
        1
        for r in promote_records
        if str(r["decision"]).split(":", 1)[1] != true_winner
    )
    disagreement_count = sum(
        1
        for r in records
        if r["naive_winner"] != str(r["decision"]).split(":", 1)[1]
        if ":" in str(r["decision"])
    )

    contaminated_winner_rates = []
    gain_lcbs = []
    for r in records:
        winner = r["final_finalist_analysis"]["winner_name"]  # type: ignore[index]
        summaries = r["summaries"]  # type: ignore[assignment]
        contaminated_winner_rates.append(summaries[winner]["contamination_rate"])  # type: ignore[index]
        gain_lcbs.append(r["final_finalist_analysis"]["lower_confidence_bound_pct"])  # type: ignore[index]

    return {
        "runs": len(records),
        "true_winner": true_winner,
        "naive_matches_true": naive_matches_true,
        "naive_false_winners": naive_false_winners,
        "final_promotes_true": final_promotes_true,
        "final_false_promotions": final_false_promotions,
        "defer_count": len(defer_records),
        "reject_count": len(reject_records),
        "promote_count": len(promote_records),
        "decision_disagreement_count": disagreement_count,
        "avg_final_winner_contamination_rate": round(mean(contaminated_winner_rates), 4),
        "avg_final_lcb_pct": round(mean(gain_lcbs), 4),
    }


def render_html(summary: dict[str, object], records: list[dict[str, object]]) -> str:
    rows = []
    for record in records:
        rows.append(
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

    summary_rows = []
    for key, value in summary.items():
        summary_rows.append(f"<tr><th>{key}</th><td>{value}</td></tr>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Simulation Summary</title>
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
  </style>
</head>
<body>
  <main>
    <h1>Noise-Aware Benchmark Simulation Summary</h1>
    <p>CPU-backed repeated-session simulation comparing naive raw-median selection against contamination-aware finalist reruns and promotion gating.</p>
    <h2>Plots</h2>
    <p><img src="plots/selection_outcomes.png" alt="Selection outcome plot" style="max-width:100%; border:1px solid #d8cfc4;"></p>
    <p><img src="plots/final_gain_vs_lcb.png" alt="Gain vs LCB plot" style="max-width:100%; border:1px solid #d8cfc4;"></p>
    <p><img src="plots/decision_timeline.png" alt="Decision timeline plot" style="max-width:100%; border:1px solid #d8cfc4;"></p>
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
        {''.join(rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def create_plots(records: list[dict[str, object]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    naive_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    gains = []
    lcbs = []
    colors = []

    for record in records:
        naive = str(record["naive_winner"])
        naive_counts[naive] = naive_counts.get(naive, 0) + 1

        decision = str(record["decision"])
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

        gains.append(record["final_finalist_analysis"]["rel_gain_pct"])  # type: ignore[index]
        lcbs.append(record["final_finalist_analysis"]["lower_confidence_bound_pct"])  # type: ignore[index]
        colors.append("#1f77b4" if decision.startswith("promote:") else "#bc6c25")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(list(naive_counts.keys()), list(naive_counts.values()), color="#b56576")
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
    statuses = []
    status_colors = []
    for record in records:
        decision = str(record["decision"])
        if decision.startswith("promote:"):
            statuses.append(2)
            winner = decision.split(":", 1)[1]
            status_colors.append("#2a9d8f" if winner == "candidate_consistent" else "#e76f51")
        else:
            statuses.append(1)
            status_colors.append("#e9c46a")

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.scatter(seeds, statuses, c=status_colors, s=70, edgecolors="black", linewidths=0.3)
    ax.set_title("Per-Run Final Outcome Timeline")
    ax.set_xlabel("Seed / Simulation Run")
    ax.set_yticks([1, 2], labels=["Defer", "Promote"])
    ax.set_ylim(0.5, 2.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "decision_timeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = BASE_DIR
    csv_path = out_dir / "simulation_results.csv"
    json_path = out_dir / "simulation_summary.json"
    html_path = out_dir / "simulation_summary.html"
    plots_dir = out_dir / "plots"

    records: list[dict[str, object]] = []
    for idx in range(args.runs):
        seed = 7 + idx
        payload = run_demo(
            backend="cpu",
            workload="matmul",
            seed=seed,
            write_report_files=False,
            verbose=False,
        )
        records.append(payload)

    summary = summarize(records, true_winner=args.true_winner)
    create_plots(records, plots_dir)

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

    json_path.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(render_html(summary, records), encoding="utf-8")

    print("Simulation complete")
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved JSON summary to {json_path}")
    print(f"Saved HTML summary to {html_path}")
    print()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
