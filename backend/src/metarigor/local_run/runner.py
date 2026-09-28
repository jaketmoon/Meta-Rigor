from __future__ import annotations

import asyncio
import hashlib
import inspect
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4
from .files import RunFolder, canonical_json
from .manifest import RunManifest
from .models import IssueRecord, StageResult, StageStatus
def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class StageIssue:
    category: str
    severity: str
    detail: str


@dataclass(frozen=True, slots=True)
class StageOutcome:
    payload: dict[str, Any]
    status: StageStatus = StageStatus.SUCCEEDED
    issues: tuple[StageIssue, ...] = ()
    human_decision_id: str | None = None
    commit: Callable[[sqlite3.Connection], None] | None = None
    raw_model_output: Any | None = None


StageExecutor = Callable[[dict[str, Any]], StageOutcome | Awaitable[StageOutcome]]


class StageRunner:
    """Single-shot canonical-input executor; no repair, attempt, or fallback."""

    def __init__(self, *, run_id: str, folder: RunFolder, manifest: RunManifest) -> None:
        self.run_id = run_id
        self.folder = folder
        self.manifest = manifest

    async def execute(
        self,
        *,
        stage: str,
        item_id: str,
        payload: dict[str, Any],
        executor: StageExecutor,
        failure_commit: Callable[[sqlite3.Connection], None] | None = None,
    ) -> StageResult:
        canonical_input = {
            "schema_version": "1.0.0",
            "run_id": self.run_id,
            "stage": stage,
            "item_id": item_id,
            "payload": payload,
        }
        input_bytes = canonical_json(canonical_input)
        input_sha256 = hashlib.sha256(input_bytes).hexdigest()
        input_path, output_path = self.folder.stage_paths(stage, item_id, input_sha256)
        input_file = self.folder.write_immutable(input_path, input_bytes)
        existing = self.manifest.exact_stage_result(self.run_id, stage, item_id, input_file.sha256)
        if existing is not None:
            if existing.status is StageStatus.RUNNING:
                raise RuntimeError(
                    "canonical Stage Result is still RUNNING; record a harness failure "
                    "or start a new Run"
                )
            if existing.output_path is None or existing.output_sha256 is None:
                raise RuntimeError("terminal Stage Result has no canonical output")
            self.folder.read_verified(existing.output_path, existing.output_sha256)
            return existing

        started_at = utc_now()
        result_id = str(uuid4())
        running = StageResult(
            result_id=result_id,
            run_id=self.run_id,
            stage=stage,
            item_id=item_id,
            input_path=input_file.path,
            input_sha256=input_file.sha256,
            output_path=None,
            output_sha256=None,
            status=StageStatus.RUNNING,
            human_decision_id=None,
            started_at=started_at,
            finished_at=None,
        )
        claimed = self.manifest.start_stage(running)
        if claimed.result_id != result_id:
            if claimed.status is StageStatus.RUNNING:
                raise RuntimeError("canonical Stage Result was already claimed")
            return claimed

        cancellation: asyncio.CancelledError | None = None
        try:
            outcome = executor(canonical_input)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if not isinstance(outcome, StageOutcome):
                raise TypeError("Stage executor must return StageOutcome")
        except asyncio.CancelledError as error:
            cancellation = error
            outcome = StageOutcome(
                payload={
                    "schema_version": "1.0.0",
                    "status": "FAILED",
                    "error": {
                        "type": "CancelledError",
                        "detail": "stage was cancelled by the execution harness",
                    },
                },
                status=StageStatus.FAILED,
                issues=(
                    StageIssue(
                        category="stage_execution_cancelled",
                        severity="ERROR",
                        detail="stage was cancelled by the execution harness",
                    ),
                ),
                commit=failure_commit,
            )
        except Exception as error:
            failure_error: dict[str, Any] = {
                "type": type(error).__name__,
                "detail": str(error)[:4000],
            }
            raw_model_output = getattr(error, "raw_model_output", None)
            if raw_model_output is not None:
                failure_error["raw_model_output"] = raw_model_output
            telemetry = getattr(error, "telemetry", None)
            if telemetry is not None:
                dump = getattr(telemetry, "model_dump", None)
                failure_error["telemetry"] = dump(mode="json") if callable(dump) else telemetry
            metrics = getattr(error, "metrics", None)
            if metrics is not None:
                failure_error["metrics"] = metrics
            error_code = getattr(error, "code", None)
            if isinstance(error_code, str):
                failure_error["code"] = error_code
            outcome = StageOutcome(
                payload={
                    "schema_version": "1.0.0",
                    "status": "FAILED",
                    "error": failure_error,
                },
                status=StageStatus.FAILED,
                issues=(
                    StageIssue(
                        category="stage_execution_failed",
                        severity="ERROR",
                        detail=f"{type(error).__name__}: {error}"[:4000],
                    ),
                ),
                commit=failure_commit,
            )
        output_payload = dict(outcome.payload)
        if outcome.raw_model_output is not None:
            output_payload["raw_model_output"] = outcome.raw_model_output
        output_bytes = canonical_json(output_payload)
        output_file = self.folder.write_immutable(output_path, output_bytes)
        finished = StageResult(
            result_id=result_id,
            run_id=self.run_id,
            stage=stage,
            item_id=item_id,
            input_path=input_file.path,
            input_sha256=input_file.sha256,
            output_path=output_file.path,
            output_sha256=output_file.sha256,
            status=outcome.status,
            human_decision_id=outcome.human_decision_id,
            started_at=started_at,
            finished_at=utc_now(),
        )
        issues = tuple(
            IssueRecord(
                issue_id=str(uuid4()),
                run_id=self.run_id,
                stage_result_id=result_id,
                item_id=item_id,
                category=issue.category,
                severity=issue.severity,
                detail=issue.detail,
                created_at=utc_now(),
            )
            for issue in outcome.issues
        )
        try:
            self.manifest.finish_stage(finished, issues=issues, commit=outcome.commit)
        except Exception as error:
            failure_payload = {
                "schema_version": "1.0.0",
                "status": "FAILED",
                "error": {
                    "type": type(error).__name__,
                    "detail": f"atomic Stage commit failed: {error}"[:4000],
                },
            }
            failure_path = output_path.removesuffix("output.json") + "commit-failure.json"
            failure_file = self.folder.write_immutable(
                failure_path, canonical_json(failure_payload)
            )
            failed = StageResult(
                result_id=result_id,
                run_id=self.run_id,
                stage=stage,
                item_id=item_id,
                input_path=input_file.path,
                input_sha256=input_file.sha256,
                output_path=failure_file.path,
                output_sha256=failure_file.sha256,
                status=StageStatus.FAILED,
                human_decision_id=None,
                started_at=started_at,
                finished_at=utc_now(),
            )
            commit_issue = IssueRecord(
                issue_id=str(uuid4()),
                run_id=self.run_id,
                stage_result_id=result_id,
                item_id=item_id,
                category="stage_commit_failed",
                severity="ERROR",
                detail=f"{type(error).__name__}: {error}"[:4000],
                created_at=utc_now(),
            )
            self.manifest.finish_stage(failed, issues=(commit_issue,))
            return failed
        if cancellation is not None:
            raise cancellation
        return finished

