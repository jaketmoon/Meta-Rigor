from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
class RunStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_HUMAN = "WAITING_HUMAN"
    SUCCEEDED = "SUCCEEDED"
    COMPLETED_WITH_ISSUES = "COMPLETED_WITH_ISSUES"
    FAILED = "FAILED"


class StageStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    WAITING_HUMAN = "WAITING_HUMAN"
    COMPLETED_WITH_ISSUES = "COMPLETED_WITH_ISSUES"


@dataclass(frozen=True, slots=True)
class StageResult:
    result_id: str
    run_id: str
    stage: str
    item_id: str
    input_path: str
    input_sha256: str
    output_path: str | None
    output_sha256: str | None
    status: StageStatus
    human_decision_id: str | None
    started_at: str
    finished_at: str | None


@dataclass(frozen=True, slots=True)
class IssueRecord:
    issue_id: str
    run_id: str
    stage_result_id: str
    item_id: str
    category: str
    severity: str
    detail: str
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class HumanDecisionRecord:
    decision_id: str
    run_id: str
    stage: str
    item_id: str
    candidate_output_sha256: str
    question: str
    status: str
    answer_json: str | None
    actor: str | None
    created_at: str
    decided_at: str | None


