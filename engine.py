from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence


@dataclass
class TelemetrySample:
    cpu_load_pct: float
    thermal_c: Optional[float]
    sm_clock_mhz: Optional[float]
    mem_clock_mhz: Optional[float]
    active_processes: Optional[int]
    launch_cpu: Optional[int]
    numa_node: Optional[int]
    synthetic_interference: bool = False


@dataclass
class TrialResult:
    candidate_name: str
    trial_index: int
    latency_ms: float
    telemetry: TelemetrySample
    label: str
    reasons: List[str] = field(default_factory=list)


@dataclass
class CandidateSummary:
    name: str
    trials: List[TrialResult]

    @property
    def acceptable_trials(self) -> List[TrialResult]:
        return [trial for trial in self.trials if trial.label == "acceptable"]

    @property
    def suspect_trials(self) -> List[TrialResult]:
        return [trial for trial in self.trials if trial.label == "suspect"]

    @property
    def contaminated_trials(self) -> List[TrialResult]:
        return [trial for trial in self.trials if trial.label == "contaminated"]

    @property
    def acceptable_count(self) -> int:
        return len(self.acceptable_trials)

    @property
    def contamination_rate(self) -> float:
        if not self.trials:
            return 0.0
        return len(self.contaminated_trials) / len(self.trials)

    @property
    def median_all_ms(self) -> Optional[float]:
        if not self.trials:
            return None
        return median(trial.latency_ms for trial in self.trials)

    @property
    def median_acceptable_ms(self) -> Optional[float]:
        if not self.acceptable_trials:
            return None
        return median(trial.latency_ms for trial in self.acceptable_trials)


@dataclass
class DecisionThresholds:
    sm_clock_drop_pct: float = 5.0
    mem_clock_drop_pct: float = 5.0
    thermal_delta_c: float = 7.0
    cpu_load_pct: float = 85.0
    min_acceptable_trials: int = 10
    promote_gain_min_pct: float = 2.0
    promote_lcb_min_pct: float = 1.5
    contamination_rate_max: float = 0.15
    finalist_contamination_penalty: float = 0.35
    finalist_min_acceptable_trials: int = 4
    base_pair_budget: int = 6
    contamination_bonus: int = 4


@dataclass
class PairwiseAnalysis:
    winner_name: str
    loser_name: str
    rel_gain_pct: float
    lower_confidence_bound_pct: float
    unresolved: bool
    reasons: List[str]


class ContaminationClassifier:
    def __init__(
        self,
        thresholds: DecisionThresholds,
        baseline_sm_clock_mhz: Optional[float] = None,
        baseline_mem_clock_mhz: Optional[float] = None,
        baseline_thermal_c: Optional[float] = None,
        allowed_cpu_set: Optional[Sequence[int]] = None,
        required_numa_node: Optional[int] = None,
        allow_shared_mode: bool = False,
    ) -> None:
        self.thresholds = thresholds
        self.baseline_sm_clock_mhz = baseline_sm_clock_mhz
        self.baseline_mem_clock_mhz = baseline_mem_clock_mhz
        self.baseline_thermal_c = baseline_thermal_c
        self.allowed_cpu_set = set(allowed_cpu_set or [])
        self.required_numa_node = required_numa_node
        self.allow_shared_mode = allow_shared_mode

    def classify(self, telemetry: TelemetrySample) -> tuple[str, List[str]]:
        reasons: List[str] = []

        if (
            self.baseline_sm_clock_mhz
            and telemetry.sm_clock_mhz
            and telemetry.sm_clock_mhz
            < self.baseline_sm_clock_mhz * (1 - self.thresholds.sm_clock_drop_pct / 100.0)
        ):
            reasons.append("sm_clock_drop")

        if (
            self.baseline_mem_clock_mhz
            and telemetry.mem_clock_mhz
            and telemetry.mem_clock_mhz
            < self.baseline_mem_clock_mhz * (1 - self.thresholds.mem_clock_drop_pct / 100.0)
        ):
            reasons.append("mem_clock_drop")

        if (
            self.baseline_thermal_c is not None
            and telemetry.thermal_c is not None
            and telemetry.thermal_c - self.baseline_thermal_c > self.thresholds.thermal_delta_c
        ):
            reasons.append("thermal_drift")

        if telemetry.cpu_load_pct > self.thresholds.cpu_load_pct:
            reasons.append("cpu_load_high")

        if telemetry.synthetic_interference:
            reasons.append("synthetic_interference")

        if (
            telemetry.active_processes is not None
            and telemetry.active_processes > 1
            and not self.allow_shared_mode
        ):
            reasons.append("concurrent_activity")

        if self.allowed_cpu_set and telemetry.launch_cpu not in self.allowed_cpu_set:
            if reasons:
                reasons.append("launch_cpu_outside_allowed_set")
            else:
                return "suspect", ["launch_cpu_outside_allowed_set"]

        if self.required_numa_node is not None and telemetry.numa_node != self.required_numa_node:
            if reasons:
                reasons.append("numa_mismatch")
            else:
                return "suspect", ["numa_mismatch"]

        if reasons:
            return "contaminated", reasons
        return "acceptable", []


class NoiseAwareBenchmarkEngine:
    def __init__(self, thresholds: Optional[DecisionThresholds] = None) -> None:
        self.thresholds = thresholds or DecisionThresholds()

    def finalist_sort_key(self, summary: CandidateSummary) -> tuple[float, float, float]:
        acceptable = summary.acceptable_count
        acceptable_med = summary.median_acceptable_ms
        if acceptable_med is None:
            return (1.0, float("inf"), float("inf"))

        evidence_penalty = 1.0 if acceptable < self.thresholds.finalist_min_acceptable_trials else 0.0
        adjusted_latency = acceptable_med * (
            1.0 + summary.contamination_rate * self.thresholds.finalist_contamination_penalty
        )
        return (evidence_penalty, adjusted_latency, acceptable_med)

    def summarize(self, trials_by_candidate: Dict[str, List[TrialResult]]) -> Dict[str, CandidateSummary]:
        return {
            name: CandidateSummary(name=name, trials=trials)
            for name, trials in trials_by_candidate.items()
        }

    def pick_finalists(self, summaries: Dict[str, CandidateSummary]) -> List[CandidateSummary]:
        ranked = sorted(
            summaries.values(),
            key=self.finalist_sort_key,
        )
        return ranked[:2]

    def naive_winner(self, summaries: Dict[str, CandidateSummary]) -> Optional[CandidateSummary]:
        ranked = sorted(
            summaries.values(),
            key=lambda summary: summary.median_all_ms
            if summary.median_all_ms is not None
            else float("inf"),
        )
        return ranked[0] if ranked else None

    def analyze_pair(self, a: CandidateSummary, b: CandidateSummary) -> PairwiseAnalysis:
        a_med = a.median_acceptable_ms
        b_med = b.median_acceptable_ms
        if a_med is None or b_med is None:
            winner, loser = (a, b) if (a_med or float("inf")) <= (b_med or float("inf")) else (b, a)
            return PairwiseAnalysis(
                winner_name=winner.name,
                loser_name=loser.name,
                rel_gain_pct=0.0,
                lower_confidence_bound_pct=-1.0,
                unresolved=True,
                reasons=["insufficient_acceptable_trials"],
            )

        winner, loser = (a, b) if a_med <= b_med else (b, a)
        winner_med = winner.median_acceptable_ms
        loser_med = loser.median_acceptable_ms
        assert winner_med is not None and loser_med is not None
        rel_gain_pct = ((loser_med - winner_med) / loser_med) * 100.0

        variability_penalty_pct = max(winner.contamination_rate, loser.contamination_rate) * 5.0
        lower_confidence_bound_pct = rel_gain_pct - variability_penalty_pct

        unresolved_reasons: List[str] = []
        if winner.acceptable_count < self.thresholds.min_acceptable_trials:
            unresolved_reasons.append(f"{winner.name}_acceptable_count_low")
        if loser.acceptable_count < self.thresholds.min_acceptable_trials:
            unresolved_reasons.append(f"{loser.name}_acceptable_count_low")
        if rel_gain_pct < self.thresholds.promote_gain_min_pct:
            unresolved_reasons.append("gain_below_threshold")
        if lower_confidence_bound_pct < self.thresholds.promote_lcb_min_pct:
            unresolved_reasons.append("lcb_below_threshold")

        return PairwiseAnalysis(
            winner_name=winner.name,
            loser_name=loser.name,
            rel_gain_pct=rel_gain_pct,
            lower_confidence_bound_pct=lower_confidence_bound_pct,
            unresolved=bool(unresolved_reasons),
            reasons=unresolved_reasons,
        )

    def rerun_budget_for_pair(self, a: CandidateSummary, b: CandidateSummary) -> int:
        budget = self.thresholds.base_pair_budget
        if a.contamination_rate > self.thresholds.contamination_rate_max:
            budget += self.thresholds.contamination_bonus
        if b.contamination_rate > self.thresholds.contamination_rate_max:
            budget += self.thresholds.contamination_bonus
        return budget

    def promotion_decision(self, analysis: PairwiseAnalysis, summaries: Dict[str, CandidateSummary]) -> str:
        winner = summaries[analysis.winner_name]
        loser = summaries[analysis.loser_name]

        if (
            analysis.rel_gain_pct >= self.thresholds.promote_gain_min_pct
            and analysis.lower_confidence_bound_pct >= self.thresholds.promote_lcb_min_pct
            and winner.acceptable_count >= self.thresholds.min_acceptable_trials
            and loser.acceptable_count >= self.thresholds.min_acceptable_trials
            and winner.contamination_rate <= self.thresholds.contamination_rate_max
            and loser.contamination_rate <= self.thresholds.contamination_rate_max * 2.0
        ):
            return f"promote:{winner.name}"

        if analysis.rel_gain_pct <= 0:
            return f"reject:{winner.name}"

        return f"defer:{analysis.winner_name}:{analysis.loser_name}"


CandidateFn = Callable[[], None]
