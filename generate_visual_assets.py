from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "visual_assets"
PLOTS_DIR = ASSETS_DIR / "plots"
DIAGRAMS_DIR = ASSETS_DIR / "diagrams"
DECK_DIR = ASSETS_DIR / "deck"
SIM_SUMMARY = BASE_DIR / "simulation_summary.json"


def load_payload() -> dict[str, object]:
    return json.loads(SIM_SUMMARY.read_text(encoding="utf-8"))


def load_records() -> list[dict[str, object]]:
    payload = load_payload()
    return payload["records"]


def load_summary() -> dict[str, object]:
    payload = load_payload()
    return payload["summary"]


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    DECK_DIR.mkdir(parents=True, exist_ok=True)


def plot_selection_outcomes(records: list[dict[str, object]]) -> None:
    naive_counts = Counter(str(r["naive_winner"]) for r in records if r["naive_winner"])
    decision_counts = Counter(str(r["decision"]) for r in records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#f6f1e8")
    for ax in axes:
        ax.set_facecolor("#fffdf9")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].bar(list(naive_counts.keys()), list(naive_counts.values()), color="#b56576", edgecolor="#3b2f2f", linewidth=0.8)
    axes[0].set_title("Naive Winner Frequency", fontsize=16, fontweight="bold")
    axes[0].set_ylabel("Runs", fontsize=12)
    axes[0].tick_params(axis="x", rotation=18, labelsize=10)

    axes[1].bar(list(decision_counts.keys()), list(decision_counts.values()), color="#5a7d2b", edgecolor="#2b2b2b", linewidth=0.8)
    axes[1].set_title("Final Decision Frequency", fontsize=16, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=18, labelsize=10)

    fig.suptitle("Selection Behavior Under Contamination-Aware Benchmarking", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS_DIR / "selection_outcomes_highres.png", dpi=280, bbox_inches="tight")
    plt.close(fig)


def plot_gain_vs_lcb(records: list[dict[str, object]]) -> None:
    gains = [float(r["final_finalist_analysis"]["rel_gain_pct"]) for r in records]  # type: ignore[index]
    lcbs = [float(r["final_finalist_analysis"]["lower_confidence_bound_pct"]) for r in records]  # type: ignore[index]
    decisions = [str(r["decision"]) for r in records]
    colors = ["#2a9d8f" if d.startswith("promote:") else "#d68c45" for d in decisions]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#f6f1e8")
    ax.set_facecolor("#fffdf9")
    ax.scatter(gains, lcbs, c=colors, s=90, alpha=0.88, edgecolors="#222222", linewidths=0.45)
    ax.axvline(2.0, color="#505050", linestyle="--", linewidth=1.3, label="Gain gate")
    ax.axhline(1.5, color="#8c2f39", linestyle="--", linewidth=1.3, label="LCB gate")
    ax.set_title("Final Pair Gain vs Lower Confidence Bound", fontsize=17, fontweight="bold")
    ax.set_xlabel("Measured gain (%)", fontsize=12)
    ax.set_ylabel("Lower confidence bound (%)", fontsize=12)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gain_vs_lcb_highres.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_decision_timeline(records: list[dict[str, object]]) -> None:
    seeds = [int(r["seed"]) for r in records]
    decisions = [str(r["decision"]) for r in records]
    statuses = [2 if d.startswith("promote:") else 1 for d in decisions]
    colors = ["#2a9d8f" if d.startswith("promote:") else "#d68c45" for d in decisions]

    fig, ax = plt.subplots(figsize=(15, 4.2), facecolor="#f6f1e8")
    ax.set_facecolor("#fffdf9")
    ax.scatter(seeds, statuses, c=colors, s=110, edgecolors="#222222", linewidths=0.45)
    ax.set_title("Per-Run Promotion Timeline", fontsize=17, fontweight="bold")
    ax.set_xlabel("Simulation run / seed", fontsize=12)
    ax.set_yticks([1, 2], labels=["Defer", "Promote"])
    ax.set_ylim(0.5, 2.5)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "decision_timeline_highres.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_svg(name: str, content: str) -> None:
    (DIAGRAMS_DIR / name).write_text(content, encoding="utf-8")


def build_diagrams() -> None:
    write_svg(
        "control_loop.svg",
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 330">
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="4" orient="auto"><path d="M0,0 L12,4 L0,8 Z" fill="#2b2b2b"/></marker>
</defs>
<rect width="1200" height="330" fill="#fffdf9"/>
<text x="50" y="40" font-family="Georgia" font-size="28" font-weight="bold" fill="#211d1a">Telemetry-Aware Selection Control Loop</text>
<g font-family="Georgia" fill="#211d1a">
  <rect x="40" y="110" rx="14" ry="14" width="180" height="80" fill="#e8f1f8" stroke="#2a6f97" stroke-width="2.4"/>
  <text x="130" y="145" text-anchor="middle" font-size="22">Kernel</text>
  <text x="130" y="171" text-anchor="middle" font-size="22">candidates</text>
  <rect x="270" y="110" rx="14" ry="14" width="190" height="80" fill="#eef5e6" stroke="#6b8e23" stroke-width="2.4"/>
  <text x="365" y="145" text-anchor="middle" font-size="22">Repeated</text>
  <text x="365" y="171" text-anchor="middle" font-size="22">benchmark trials</text>
  <rect x="510" y="65" rx="14" ry="14" width="220" height="70" fill="#f9f1e5" stroke="#bc6c25" stroke-width="2.4"/>
  <text x="620" y="105" text-anchor="middle" font-size="20">Telemetry capture</text>
  <rect x="510" y="160" rx="14" ry="14" width="220" height="70" fill="#fce8ea" stroke="#8c2f39" stroke-width="2.4"/>
  <text x="620" y="200" text-anchor="middle" font-size="20">Contamination state</text>
  <rect x="790" y="110" rx="14" ry="14" width="180" height="80" fill="#e8f1f8" stroke="#2a6f97" stroke-width="2.4"/>
  <text x="880" y="145" text-anchor="middle" font-size="22">Finalist</text>
  <text x="880" y="171" text-anchor="middle" font-size="22">pair analysis</text>
  <rect x="1020" y="110" rx="14" ry="14" width="150" height="80" fill="#eef5e6" stroke="#6b8e23" stroke-width="2.4"/>
  <text x="1095" y="145" text-anchor="middle" font-size="20">Promote /</text>
  <text x="1095" y="171" text-anchor="middle" font-size="20">Defer / Reject</text>
</g>
<g stroke="#2b2b2b" stroke-width="2.4" fill="none" marker-end="url(#arrow)">
  <line x1="220" y1="150" x2="270" y2="150"/>
  <line x1="460" y1="150" x2="510" y2="100"/>
  <line x1="460" y1="150" x2="510" y2="195"/>
  <line x1="730" y1="195" x2="790" y2="150"/>
  <line x1="970" y1="150" x2="1020" y2="150"/>
  <path d="M1020 225 C900 300, 520 300, 365 195"/>
</g>
<text x="760" y="305" font-family="Georgia" font-size="18" fill="#8c2f39">Only unresolved finalist pairs receive extra reruns</text>
</svg>""",
    )
    write_svg(
        "gpu_batch_flow.svg",
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 420">
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="4" orient="auto"><path d="M0,0 L12,4 L0,8 Z" fill="#2b2b2b"/></marker>
</defs>
<rect width="1200" height="420" fill="#fffdf9"/>
<text x="50" y="42" font-family="Georgia" font-size="28" font-weight="bold" fill="#211d1a">GPU Batch Execution Flow</text>
<g font-family="Georgia" fill="#211d1a">
  <rect x="50" y="110" width="180" height="74" rx="14" fill="#e8f1f8" stroke="#2a6f97" stroke-width="2.4"/>
  <text x="140" y="154" text-anchor="middle" font-size="22">setup_gpu.py</text>
  <rect x="280" y="110" width="220" height="74" rx="14" fill="#eef5e6" stroke="#6b8e23" stroke-width="2.4"/>
  <text x="390" y="154" text-anchor="middle" font-size="22">demo.py --backend triton</text>
  <rect x="550" y="110" width="260" height="74" rx="14" fill="#f9f1e5" stroke="#bc6c25" stroke-width="2.4"/>
  <text x="680" y="144" text-anchor="middle" font-size="22">triton_batch_runner.py</text>
  <text x="680" y="168" text-anchor="middle" font-size="18">generated matmul variants</text>
  <rect x="860" y="110" width="270" height="74" rx="14" fill="#fce8ea" stroke="#8c2f39" stroke-width="2.4"/>
  <text x="995" y="144" text-anchor="middle" font-size="22">variant_search_report.py</text>
  <text x="995" y="168" text-anchor="middle" font-size="18">refresh HTML summary</text>

  <rect x="280" y="250" width="220" height="88" rx="14" fill="#fffaf2" stroke="#6c6258" stroke-width="2.1"/>
  <text x="390" y="286" text-anchor="middle" font-size="20">latest_run.html</text>
  <text x="390" y="310" text-anchor="middle" font-size="18">single-run artifact</text>

  <rect x="550" y="250" width="260" height="88" rx="14" fill="#fffaf2" stroke="#6c6258" stroke-width="2.1"/>
  <text x="680" y="286" text-anchor="middle" font-size="20">batch_outputs/...</text>
  <text x="680" y="310" text-anchor="middle" font-size="18">CSV, JSON, HTML, PNG</text>

  <rect x="860" y="250" width="270" height="88" rx="14" fill="#fffaf2" stroke="#6c6258" stroke-width="2.1"/>
  <text x="995" y="286" text-anchor="middle" font-size="20">variant_search_report.html</text>
  <text x="995" y="310" text-anchor="middle" font-size="18">candidate table + results</text>
</g>
<g stroke="#2b2b2b" stroke-width="2.4" fill="none" marker-end="url(#arrow)">
  <line x1="230" y1="147" x2="280" y2="147"/>
  <line x1="500" y1="147" x2="550" y2="147"/>
  <line x1="810" y1="147" x2="860" y2="147"/>
  <line x1="390" y1="184" x2="390" y2="250"/>
  <line x1="680" y1="184" x2="680" y2="250"/>
  <line x1="995" y1="184" x2="995" y2="250"/>
</g>
</svg>""",
    )
    write_svg(
        "generated_variant_map.svg",
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 520">
<rect width="1200" height="520" fill="#fffdf9"/>
<text x="50" y="42" font-family="Georgia" font-size="28" font-weight="bold" fill="#211d1a">Generated Triton Matmul Variant Map</text>
<g font-family="Georgia" fill="#211d1a">
  <rect x="60" y="90" width="250" height="150" rx="16" fill="#e8f1f8" stroke="#2a6f97" stroke-width="2.4"/>
  <text x="185" y="130" text-anchor="middle" font-size="22">balanced</text>
  <text x="185" y="164" text-anchor="middle" font-size="18">128 x 128 x 64</text>
  <text x="185" y="194" text-anchor="middle" font-size="18">warps: 4, stages: 4</text>
  <text x="185" y="224" text-anchor="middle" font-size="18">general-purpose square tile</text>

  <rect x="335" y="90" width="250" height="150" rx="16" fill="#eef5e6" stroke="#6b8e23" stroke-width="2.4"/>
  <text x="460" y="130" text-anchor="middle" font-size="22">wide_n</text>
  <text x="460" y="164" text-anchor="middle" font-size="18">64 x 256 x 32</text>
  <text x="460" y="194" text-anchor="middle" font-size="18">warps: 4, stages: 4</text>
  <text x="460" y="224" text-anchor="middle" font-size="18">favours larger N dimension</text>

  <rect x="610" y="90" width="250" height="150" rx="16" fill="#f9f1e5" stroke="#bc6c25" stroke-width="2.4"/>
  <text x="735" y="130" text-anchor="middle" font-size="22">aggressive</text>
  <text x="735" y="164" text-anchor="middle" font-size="18">128 x 256 x 64</text>
  <text x="735" y="194" text-anchor="middle" font-size="18">warps: 8, stages: 3</text>
  <text x="735" y="224" text-anchor="middle" font-size="18">higher-throughput regime</text>

  <rect x="885" y="90" width="250" height="150" rx="16" fill="#fce8ea" stroke="#8c2f39" stroke-width="2.4"/>
  <text x="1010" y="130" text-anchor="middle" font-size="22">compact</text>
  <text x="1010" y="164" text-anchor="middle" font-size="18">64 x 64 x 32</text>
  <text x="1010" y="194" text-anchor="middle" font-size="18">warps: 2, stages: 5</text>
  <text x="1010" y="224" text-anchor="middle" font-size="18">lighter-weight execution</text>

  <rect x="190" y="320" width="820" height="110" rx="16" fill="#fffaf2" stroke="#6c6258" stroke-width="2.2"/>
  <text x="600" y="360" text-anchor="middle" font-size="22">All variants flow into the same contamination-aware benchmark engine</text>
  <text x="600" y="392" text-anchor="middle" font-size="18">Telemetry capture -> contamination classification -> finalist pair reruns -> promotion gate</text>
</g>
</svg>""",
    )


def build_deck_assets(summary: dict[str, object]) -> None:
    fig = plt.figure(figsize=(14, 8), facecolor="#f3eee6")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(0.05, 0.92, "Reliable GPU Kernel Bench", fontsize=28, fontweight="bold", family="serif", color="#211d1a")
    ax.text(
        0.05,
        0.865,
        "Telemetry-aware contamination filtering, finalist-pair reruns, and confidence-gated promotion.",
        fontsize=15,
        family="serif",
        color="#5b534b",
    )

    cards = [
        ("Simulation runs", str(summary["runs"]), "#e8f1f8", "#2a6f97"),
        ("Naive false winners", str(summary["naive_false_winners"]), "#fce8ea", "#8c2f39"),
        ("Correct promotions", str(summary["final_promotes_true"]), "#eef5e6", "#5a7d2b"),
        ("False promotions", str(summary["final_false_promotions"]), "#fff4dd", "#bc6c25"),
    ]
    x_positions = [0.05, 0.285, 0.52, 0.755]
    for (label, value, fill, edge), x in zip(cards, x_positions):
        rect = plt.Rectangle((x, 0.67), 0.19, 0.15, facecolor=fill, edgecolor=edge, linewidth=2.2)
        ax.add_patch(rect)
        ax.text(x + 0.095, 0.765, value, ha="center", va="center", fontsize=30, fontweight="bold", family="serif", color="#211d1a")
        ax.text(x + 0.095, 0.705, label, ha="center", va="center", fontsize=12.5, family="serif", color="#5b534b")

    rect = plt.Rectangle((0.05, 0.12), 0.90, 0.46, facecolor="#fffdf9", edgecolor="#d8cfc4", linewidth=1.8)
    ax.add_patch(rect)
    ax.text(0.08, 0.52, "Key prototype story", fontsize=18, fontweight="bold", family="serif", color="#211d1a")
    bullet_lines = [
        "Naive raw-median selection is consistently misled by contaminated benchmark sessions.",
        "The contamination-aware loop routes uncertainty into finalist-only reruns instead of brute-force retesting.",
        "Promotion is gated on evidence quality, which trades some throughput for far fewer false wins.",
        "Generated Triton matmul variants are now wired into the same benchmark and promotion engine.",
    ]
    y = 0.46
    for line in bullet_lines:
        ax.text(0.09, y, f"- {line}", fontsize=15, family="serif", color="#2b2b2b")
        y -= 0.08

    ax.text(0.08, 0.18, "Use this card in slides, demos, or repo overviews.", fontsize=13, family="serif", color="#6c6258")
    fig.savefig(DECK_DIR / "benchmark_story_card.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    (DECK_DIR / "control_loop_presentation.svg").write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 520">
<defs>
  <marker id="arrow" markerWidth="14" markerHeight="14" refX="12" refY="5" orient="auto"><path d="M0,0 L14,5 L0,10 Z" fill="#2b2b2b"/></marker>
</defs>
<rect width="1600" height="520" fill="#fffdf9"/>
<text x="70" y="60" font-family="Georgia" font-size="42" font-weight="bold" fill="#211d1a">Telemetry-Aware Kernel Selection</text>
<text x="70" y="98" font-family="Georgia" font-size="22" fill="#5b534b">Closed-loop rerun allocation and confidence-gated promotion</text>
<g font-family="Georgia" fill="#211d1a">
  <rect x="70" y="190" rx="20" ry="20" width="240" height="110" fill="#e8f1f8" stroke="#2a6f97" stroke-width="3"/>
  <text x="190" y="236" text-anchor="middle" font-size="30">Candidate</text>
  <text x="190" y="272" text-anchor="middle" font-size="30">kernels</text>
  <rect x="390" y="190" rx="20" ry="20" width="280" height="110" fill="#eef5e6" stroke="#6b8e23" stroke-width="3"/>
  <text x="530" y="236" text-anchor="middle" font-size="30">Repeated trials</text>
  <text x="530" y="272" text-anchor="middle" font-size="24">with telemetry capture</text>
  <rect x="760" y="120" rx="20" ry="20" width="300" height="96" fill="#f9f1e5" stroke="#bc6c25" stroke-width="3"/>
  <text x="910" y="175" text-anchor="middle" font-size="28">Contamination signals</text>
  <rect x="760" y="250" rx="20" ry="20" width="300" height="96" fill="#fce8ea" stroke="#8c2f39" stroke-width="3"/>
  <text x="910" y="305" text-anchor="middle" font-size="28">Pairwise ambiguity</text>
  <rect x="1160" y="190" rx="20" ry="20" width="300" height="110" fill="#e8f1f8" stroke="#2a6f97" stroke-width="3"/>
  <text x="1310" y="236" text-anchor="middle" font-size="30">Finalist-only</text>
  <text x="1310" y="272" text-anchor="middle" font-size="30">reruns + promotion gate</text>
</g>
<g stroke="#2b2b2b" stroke-width="3" fill="none" marker-end="url(#arrow)">
  <line x1="310" y1="245" x2="390" y2="245"/>
  <line x1="670" y1="245" x2="760" y2="170"/>
  <line x1="670" y1="245" x2="760" y2="298"/>
  <line x1="1060" y1="298" x2="1160" y2="245"/>
</g>
<path d="M1160 360 C1010 455, 640 455, 530 315" fill="none" stroke="#8c2f39" stroke-width="3.2" marker-end="url(#arrow)"/>
<text x="900" y="450" font-family="Georgia" font-size="24" fill="#8c2f39">Only unresolved finalist pairs consume added benchmark budget</text>
</svg>""",
        encoding="utf-8",
    )


def build_gallery() -> None:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Visual Asset Gallery</title>
  <style>
    body { margin: 0; background: #f3eee6; color: #211d1a; font-family: Georgia, "Times New Roman", serif; }
    main { max-width: 1180px; margin: 24px auto; padding: 30px 36px 44px; background: #fffdf8; border: 1px solid #d8cfc4; box-shadow: 0 16px 40px rgba(0,0,0,.08); }
    h1, h2 { margin-top: 0; }
    h2 { margin-top: 28px; padding-top: 10px; border-top: 1px solid #d8cfc4; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    .card { border: 1px solid #d8cfc4; background: #fffaf2; padding: 14px 16px; }
    img { max-width: 100%; border: 1px solid #d8cfc4; background: white; }
    a { color: #2a6f97; }
    p { line-height: 1.45; }
  </style>
</head>
<body>
  <main>
    <h1>Visual Asset Gallery</h1>
    <p>Presentation-ready diagrams and higher-quality plots for the benchmark-selection prototype.</p>
    <h2>Diagrams</h2>
    <div class="grid">
      <div class="card"><h3><a href="diagrams/control_loop.svg">Control Loop</a></h3><img src="diagrams/control_loop.svg" alt="Control loop"></div>
      <div class="card"><h3><a href="diagrams/gpu_batch_flow.svg">GPU Batch Flow</a></h3><img src="diagrams/gpu_batch_flow.svg" alt="GPU batch flow"></div>
      <div class="card"><h3><a href="diagrams/generated_variant_map.svg">Generated Variant Map</a></h3><img src="diagrams/generated_variant_map.svg" alt="Generated variant map"></div>
    </div>
    <h2>Deck Assets</h2>
    <div class="grid">
      <div class="card"><h3><a href="deck/benchmark_story_card.png">Benchmark Story Card</a></h3><img src="deck/benchmark_story_card.png" alt="Benchmark story card"></div>
      <div class="card"><h3><a href="deck/control_loop_presentation.svg">Presentation Control Loop</a></h3><img src="deck/control_loop_presentation.svg" alt="Presentation control loop"></div>
    </div>
    <h2>Plots</h2>
    <div class="grid">
      <div class="card"><h3><a href="plots/selection_outcomes_highres.png">Selection Outcomes</a></h3><img src="plots/selection_outcomes_highres.png" alt="Selection outcomes"></div>
      <div class="card"><h3><a href="plots/gain_vs_lcb_highres.png">Gain vs LCB</a></h3><img src="plots/gain_vs_lcb_highres.png" alt="Gain vs LCB"></div>
      <div class="card"><h3><a href="plots/decision_timeline_highres.png">Decision Timeline</a></h3><img src="plots/decision_timeline_highres.png" alt="Decision timeline"></div>
    </div>
  </main>
</body>
</html>"""
    (ASSETS_DIR / "visual_gallery.html").write_text(html, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    records = load_records()
    summary = load_summary()
    plot_selection_outcomes(records)
    plot_gain_vs_lcb(records)
    plot_decision_timeline(records)
    build_diagrams()
    build_deck_assets(summary)
    build_gallery()
    print(f"Saved visual assets to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
