from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal
from .models import Certainty, DomainJudgment
_JUDGMENT_BOUNDS: dict[DomainJudgment, tuple[int, int]] = {
    "NOT_SERIOUS": (0, 0),
    "BORDERLINE_NOT_SERIOUS_SERIOUS": (0, 1),
    "SERIOUS": (1, 1),
    "BORDERLINE_SERIOUS_VERY_SERIOUS": (1, 2),
    "VERY_SERIOUS": (2, 2),
}


_CERTAINTY_BY_DOWNGRADE: tuple[Certainty, ...] = (
    "HIGH",
    "MODERATE",
    "LOW",
    "VERY_LOW",
)


@dataclass(frozen=True, slots=True)
class BinaryImprecisionDecision:
    """冻结的 binary RR/OR imprecision decision table 输出。"""

    judgment: DomainJudgment
    decision: str
    relative_effect_change_percent: float
    ois_required_participants: int | None
    ois_met: bool | None
    ois_reason: str
    issues: tuple[str, ...]


def binary_imprecision_decision(
    *,
    effect_measure: Literal["RR", "OR"],
    estimate: float,
    ci_lower: float,
    ci_upper: float,
    participant_count: int | None,
    control_event_count: int | None,
    control_participant_count: int | None,
    threshold_value: float = 1.0,
) -> BinaryImprecisionDecision:
    """统一执行正式 EC 与 benchmark 共用的 binary imprecision 规则。"""

    if not 0 < ci_lower <= estimate <= ci_upper:
        raise ValueError("binary imprecision requires a positive ordered confidence interval")
    if participant_count is not None and participant_count < 1:
        raise ValueError("binary imprecision participant_count must be positive")
    if (control_event_count is None) != (control_participant_count is None):
        raise ValueError("binary imprecision control risk requires both numerator and denominator")
    baseline = None
    if control_event_count is not None and control_participant_count is not None:
        if (
            control_participant_count < 1
            or not 0 <= control_event_count <= control_participant_count
        ):
            raise ValueError("binary imprecision control risk is invalid")
        baseline = control_event_count / control_participant_count
    relative_effect = estimate
    if effect_measure == "OR" and baseline is not None:
        relative_effect = estimate / ((1 - baseline) + baseline * estimate)
    relative_change = abs(relative_effect - 1) * 100
    ois_required = _binary_ois(baseline, relative_effect)
    if relative_change < 30:
        ois_reason = "relative effect change 小于 30%，不触发 Core GRADE large-effect OIS path。"
    elif ois_required is None:
        ois_reason = "effect 可能超过 30%，但缺少可核验 control risk，OIS 保持未知。"
    else:
        ois_reason = (
            "effect 可能超过 30%；按 alpha=.05、beta=.20、20% modest relative change "
            f"计算 OIS={ois_required}，observed N={participant_count}。"
        )
    ois_met = (
        participant_count >= ois_required
        if participant_count is not None and ois_required is not None and relative_change >= 30
        else None
    )
    issues: tuple[str, ...] = ()
    if ci_lower <= threshold_value <= ci_upper:
        judgment: DomainJudgment = "SERIOUS"
        decision = "CI 跨越冻结 certainty threshold，降一级。"
    elif relative_change < 30:
        judgment = "NOT_SERIOUS"
        decision = "point estimate 的相对变化小于 30%，不触发 large-effect/OIS path。"
    elif relative_change <= 40:
        if ois_met is True:
            judgment = "NOT_SERIOUS"
            decision = "point estimate 位于 30%–40% close-call 带，且 OIS 已满足。"
        else:
            judgment = "BORDERLINE_NOT_SERIOUS_SERIOUS"
            decision = "point estimate 位于 30%–40% close-call 带，OIS 未满足或未知。"
            issues = (ois_reason,)
    elif ois_met is True:
        judgment = "NOT_SERIOUS"
        decision = "point estimate 的相对变化超过 40%，但 OIS 已满足。"
    else:
        judgment = "SERIOUS"
        decision = "point estimate 的相对变化超过 40%，且 OIS 未满足或未知。"
        issues = (ois_reason,)
    return BinaryImprecisionDecision(
        judgment=judgment,
        decision=decision,
        relative_effect_change_percent=round(relative_change, 8),
        ois_required_participants=ois_required,
        ois_met=ois_met,
        ois_reason=ois_reason,
        issues=issues,
    )


def judgment_bounds(judgment: DomainJudgment) -> tuple[int, int]:
    return _JUDGMENT_BOUNDS[judgment]


def final_certainty(overall_downgrade_levels: int) -> Certainty:
    if not 0 <= overall_downgrade_levels <= 3:
        raise ValueError("overall downgrade must be between zero and three")
    return _CERTAINTY_BY_DOWNGRADE[overall_downgrade_levels]


def _binary_ois(
    control_risk: float | None,
    observed_relative_effect: float | None,
) -> int | None:
    """Core GRADE 2025 large-effect 路径的双侧两组 OIS（20% modest relative effect）。"""

    if control_risk is None or observed_relative_effect is None or not 0 < control_risk < 1:
        return None
    intervention_risk = (
        control_risk * 0.8 if observed_relative_effect < 1 else min(control_risk * 1.2, 1 - 1e-12)
    )
    difference = abs(intervention_risk - control_risk)
    if difference == 0:
        return None
    average = (control_risk + intervention_risk) / 2
    z_alpha = NormalDist().inv_cdf(0.975)
    z_beta = NormalDist().inv_cdf(0.8)
    numerator = (
        z_alpha * math.sqrt(2 * average * (1 - average))
        + z_beta
        * math.sqrt(control_risk * (1 - control_risk) + intervention_risk * (1 - intervention_risk))
    ) ** 2
    return math.ceil(2 * numerator / (difference**2))


