from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from .models import HumanDecisionRecord, IssueRecord, RunStatus, StageResult, StageStatus
_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    specialist TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN (
        'CREATED','RUNNING','WAITING_HUMAN','SUCCEEDED','COMPLETED_WITH_ISSUES','FAILED'
    )),
    input_path TEXT NOT NULL,
    input_sha256 TEXT NOT NULL CHECK (length(input_sha256) = 64),
    final_output_path TEXT,
    final_output_sha256 TEXT CHECK (
        final_output_sha256 IS NULL OR length(final_output_sha256) = 64
    ),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    CHECK ((status IN ('CREATED','RUNNING','WAITING_HUMAN') AND finished_at IS NULL)
        OR (status IN ('SUCCEEDED','COMPLETED_WITH_ISSUES','FAILED') AND finished_at IS NOT NULL)),
    CHECK (status NOT IN ('SUCCEEDED','COMPLETED_WITH_ISSUES')
        OR (final_output_path IS NOT NULL AND final_output_sha256 IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS stage_results (
    result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    item_id TEXT NOT NULL,
    input_path TEXT NOT NULL,
    input_sha256 TEXT NOT NULL CHECK (length(input_sha256) = 64),
    output_path TEXT,
    output_sha256 TEXT CHECK (output_sha256 IS NULL OR length(output_sha256) = 64),
    status TEXT NOT NULL CHECK (status IN (
        'RUNNING','SUCCEEDED','FAILED','WAITING_HUMAN','COMPLETED_WITH_ISSUES'
    )),
    human_decision_id TEXT REFERENCES human_decisions(decision_id),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    UNIQUE (run_id, stage, item_id, input_sha256),
    CHECK ((status = 'RUNNING' AND finished_at IS NULL)
        OR (status <> 'RUNNING' AND finished_at IS NOT NULL)),
    CHECK ((output_path IS NULL) = (output_sha256 IS NULL))
);

CREATE TABLE IF NOT EXISTS issues (
    issue_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    stage_result_id TEXT NOT NULL REFERENCES stage_results(result_id) ON DELETE CASCADE,
    item_id TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('INFO','WARNING','ERROR')),
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS human_decisions (
    decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    item_id TEXT NOT NULL,
    candidate_output_sha256 TEXT NOT NULL CHECK (length(candidate_output_sha256) = 64),
    question TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING','ACCEPTED','REJECTED','PROVIDED_INPUT')),
    answer_json TEXT,
    actor TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    CHECK ((status = 'PENDING' AND answer_json IS NULL AND actor IS NULL AND decided_at IS NULL)
        OR (status <> 'PENDING' AND answer_json IS NOT NULL AND actor IS NOT NULL
            AND decided_at IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS ix_stage_results_run_stage ON stage_results(run_id, stage, item_id);
CREATE INDEX IF NOT EXISTS ix_issues_run_stage ON issues(run_id, stage_result_id);
"""


class RunManifest:
    """Small authoritative SQLite index; does not store large model or source content."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(_SCHEMA)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def transaction(self):
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_run(
        self,
        *,
        run_id: str,
        specialist: str,
        input_path: str,
        input_sha256: str,
        started_at: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO runs (
                    run_id, specialist, status, input_path, input_sha256, started_at
                ) VALUES (?, ?, 'CREATED', ?, ?, ?)""",
                (run_id, specialist, input_path, input_sha256, started_at),
            )

    def set_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        finished_at: str | None = None,
        final_output_path: str | None = None,
        final_output_sha256: str | None = None,
    ) -> None:
        with self.transaction() as connection:
            changed = connection.execute(
                """UPDATE runs SET status = ?, finished_at = ?,
                    final_output_path = COALESCE(?, final_output_path),
                    final_output_sha256 = COALESCE(?, final_output_sha256)
                    WHERE run_id = ?
                      AND status NOT IN ('SUCCEEDED','COMPLETED_WITH_ISSUES','FAILED')""",
                (
                    status.value,
                    finished_at,
                    final_output_path,
                    final_output_sha256,
                    run_id,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError(f"Run not found or already terminal: {run_id}")

    def start_stage(self, result: StageResult) -> StageResult:
        if result.status is not StageStatus.RUNNING or result.finished_at is not None:
            raise ValueError("start_stage requires a RUNNING result")
        with self.transaction() as connection:
            existing = connection.execute(
                """SELECT * FROM stage_results
                   WHERE run_id = ? AND stage = ? AND item_id = ? AND input_sha256 = ?""",
                (result.run_id, result.stage, result.item_id, result.input_sha256),
            ).fetchone()
            if existing is not None:
                return self._stage(existing)
            connection.execute(
                """INSERT INTO stage_results (
                    result_id, run_id, stage, item_id, input_path, input_sha256,
                    output_path, output_sha256, status, human_decision_id,
                    started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.result_id,
                    result.run_id,
                    result.stage,
                    result.item_id,
                    result.input_path,
                    result.input_sha256,
                    result.output_path,
                    result.output_sha256,
                    result.status.value,
                    result.human_decision_id,
                    result.started_at,
                    result.finished_at,
                ),
            )
        return result

    def finish_stage(
        self,
        result: StageResult,
        *,
        issues: Iterable[IssueRecord] = (),
        human_decision: HumanDecisionRecord | None = None,
        commit: Callable[[sqlite3.Connection], None] | None = None,
    ) -> None:
        if result.status is StageStatus.RUNNING or result.finished_at is None:
            raise ValueError("finish_stage requires a terminal result")
        if result.output_path is None or result.output_sha256 is None:
            raise ValueError("terminal stage result requires a canonical output")
        with self.transaction() as connection:
            current = connection.execute(
                "SELECT status FROM stage_results WHERE result_id = ?",
                (result.result_id,),
            ).fetchone()
            if current is None:
                raise KeyError(f"Stage result not found: {result.result_id}")
            if current["status"] != StageStatus.RUNNING.value:
                if current["status"] == result.status.value:
                    return
                raise ValueError("terminal Stage Result cannot be overwritten")
            if human_decision is not None:
                if (
                    result.human_decision_id != human_decision.decision_id
                    or result.run_id != human_decision.run_id
                    or result.stage != human_decision.stage
                    or result.item_id != human_decision.item_id
                ):
                    raise ValueError("Human Decision is outside its Stage Result")
                self._insert_human_decision(connection, human_decision)
            if commit is not None:
                commit(connection)
            connection.execute(
                """UPDATE stage_results SET output_path = ?, output_sha256 = ?, status = ?,
                    human_decision_id = ?, finished_at = ? WHERE result_id = ?""",
                (
                    result.output_path,
                    result.output_sha256,
                    result.status.value,
                    result.human_decision_id,
                    result.finished_at,
                    result.result_id,
                ),
            )
            for issue in issues:
                if issue.stage_result_id != result.result_id or issue.run_id != result.run_id:
                    raise ValueError("Issue is outside its Stage Result")
                connection.execute(
                    """INSERT INTO issues (
                        issue_id, run_id, stage_result_id, item_id, category, severity,
                        detail, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        issue.issue_id,
                        issue.run_id,
                        issue.stage_result_id,
                        issue.item_id,
                        issue.category,
                        issue.severity,
                        issue.detail,
                        issue.created_at,
                    ),
                )

    def exact_stage_result(
        self, run_id: str, stage: str, item_id: str, input_sha256: str
    ) -> StageResult | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM stage_results
                   WHERE run_id = ? AND stage = ? AND item_id = ? AND input_sha256 = ?""",
                (run_id, stage, item_id, input_sha256),
            ).fetchone()
        return None if row is None else self._stage(row)

    def list_stage_results(self, run_id: str) -> tuple[StageResult, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM stage_results WHERE run_id = ? ORDER BY started_at, result_id",
                (run_id,),
            ).fetchall()
        return tuple(self._stage(row) for row in rows)

    def create_human_decision(self, decision: HumanDecisionRecord) -> None:
        with self.transaction() as connection:
            self._insert_human_decision(connection, decision)

    @staticmethod
    def _insert_human_decision(
        connection: sqlite3.Connection, decision: HumanDecisionRecord
    ) -> None:
        connection.execute(
            """INSERT INTO human_decisions (
                decision_id, run_id, stage, item_id, candidate_output_sha256,
                question, status, answer_json, actor, created_at, decided_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision.decision_id,
                decision.run_id,
                decision.stage,
                decision.item_id,
                decision.candidate_output_sha256,
                decision.question,
                decision.status,
                decision.answer_json,
                decision.actor,
                decision.created_at,
                decision.decided_at,
            ),
        )

    def add_issue(self, issue: IssueRecord) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO issues (
                    issue_id, run_id, stage_result_id, item_id, category, severity,
                    detail, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    issue.issue_id,
                    issue.run_id,
                    issue.stage_result_id,
                    issue.item_id,
                    issue.category,
                    issue.severity,
                    issue.detail,
                    issue.created_at,
                ),
            )

    def issue_count(self, run_id: str) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT count(*) AS value FROM issues WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row["value"])

    def resolve_human_decision(
        self,
        decision_id: str,
        *,
        candidate_output_sha256: str,
        status: str,
        answer: dict[str, Any],
        actor: str,
        decided_at: str,
    ) -> None:
        if status not in {"ACCEPTED", "REJECTED", "PROVIDED_INPUT"}:
            raise ValueError("Human Decision resolution status is invalid")
        with self.transaction() as connection:
            changed = connection.execute(
                """UPDATE human_decisions SET status = ?, answer_json = ?, actor = ?, decided_at = ?
                   WHERE decision_id = ? AND status = 'PENDING'
                     AND candidate_output_sha256 = ?""",
                (
                    status,
                    json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    actor,
                    decided_at,
                    decision_id,
                    candidate_output_sha256,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError(
                    "Human Decision is missing, already resolved, or candidate changed"
                )

    @staticmethod
    def _stage(row: sqlite3.Row) -> StageResult:
        return StageResult(
            result_id=row["result_id"],
            run_id=row["run_id"],
            stage=row["stage"],
            item_id=row["item_id"],
            input_path=row["input_path"],
            input_sha256=row["input_sha256"],
            output_path=row["output_path"],
            output_sha256=row["output_sha256"],
            status=StageStatus(row["status"]),
            human_decision_id=row["human_decision_id"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

