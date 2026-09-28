from __future__ import annotations

import asyncio
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4
from metarigor.local_run import RunFolder, RunManifest
from metarigor.schema_agent import SchemaAgentOutputError, SchemaAgentRunner
from .definitions import publication_judge_template
from .judge import CRITERIA, RATING_ANCHORS
from .models import FactJudgeEvidence, JudgeCriterionId, canonical_json
from .publication_coverage import decide_headlines
from .publication_judge_models import PublicationClaimLineageV2, PublicationHeadlineDecision, PublicationJudgeArtifactFile, PublicationJudgeClaim, PublicationJudgeCriterionAgentOutput, PublicationJudgeCriterionArtifacts, PublicationJudgeCriterionInput, PublicationJudgeCriterionOutput, PublicationJudgeEvidence, PublicationJudgeRatingAnchor, PublicationJudgeStructuredScore, PublicationJudgeSurface, PublicationJudgeValidationReport, PublicationWriterFact, PublicationWriterProjection
from .publication_models import ManuscriptDeliveryV2, ManuscriptFactPackageV2, ManuscriptRunResultV2
def native_judge_template(
    criterion, *, score_mode, structured_score=False, compact_format=False, array_score=False
):
    """Native candidates retain historical nonempty Fact bindings; permissive external-wrapper inputs do not propagate."""
    template = publication_judge_template(
        criterion,
        score_mode=score_mode,
        structured_score=structured_score,
        compact_format=compact_format,
        array_score=array_score,
    )
    schema = deepcopy(template.input_schema)
    claim = schema["$defs"]["PublicationJudgeClaim"]
    claim["properties"]["fact_ids"].pop("default", None)
    claim["properties"]["fact_ids"]["minItems"] = 1
    if "fact_ids" not in claim["required"]:
        claim["required"].insert(2, "fact_ids")
    return replace(template, input_schema=schema)


@dataclass(frozen=True, slots=True)
class PublicationManuscriptJudgeResult:
    validation_id: str
    status: str
    validation_root: Path
    report_path: Path
    weighted_score: float | None
    critical_issue_count: int | None = None


class PublicationJudgePreflightError(ValueError):
    """Candidate preflight failure; the exception retains the location of sealed failure evidence."""

    def __init__(
        self,
        *,
        validation_id: str,
        validation_root: Path,
        failure_path: Path,
        cause: Exception,
    ) -> None:
        self.validation_id = validation_id
        self.validation_root = validation_root
        self.failure_path = failure_path
        super().__init__(
            f"{type(cause).__name__}: {cause}; preflight failure preserved at {failure_path}"
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    run_id: str
    package: ManuscriptFactPackageV2
    projection: PublicationWriterProjection
    delivery: ManuscriptDeliveryV2
    delivery_path: str
    delivery_sha256: str
    fact_package_sha256: str
    journal_profile_sha256: str
    projection_sha256: str
    lineage_sha256: str
    claims: tuple[PublicationJudgeClaim, ...]
    surfaces: dict[str, PublicationJudgeSurface]
    external_mode: bool
    candidate_origin: str


_SURFACE_ROLES: dict[JudgeCriterionId, tuple[str, ...]] = {
    "FACTUAL_FIDELITY": (
        "MANUSCRIPT",
        "TABLES",
        "SUPPLEMENT",
        "REFERENCES",
        "PRISMA_FLOW",
        "FOREST_PLOT",
    ),
    "PROTOCOL_AND_CONDUCT_FIDELITY": ("MANUSCRIPT", "SUPPLEMENT", "PRISMA_FLOW"),
    "MATERIAL_COVERAGE": (
        "MANUSCRIPT",
        "TABLES",
        "SUPPLEMENT",
        "REFERENCES",
        "PRISMA_FLOW",
        "FOREST_PLOT",
    ),
    "INTERPRETATION_CALIBRATION": ("MANUSCRIPT", "TABLES", "FOREST_PLOT"),
    "SOURCE_TRACEABILITY": ("MANUSCRIPT", "TABLES", "SUPPLEMENT", "REFERENCES"),
    "CROSS_SECTION_CONSISTENCY": ("MANUSCRIPT", "TABLES", "SUPPLEMENT", "FOREST_PLOT"),
    "SCIENTIFIC_ORGANIZATION": ("MANUSCRIPT",),
}


_FACT_KINDS: dict[JudgeCriterionId, set[str] | None] = {
    "FACTUAL_FIDELITY": None,
    "PROTOCOL_AND_CONDUCT_FIDELITY": {
        "NARRATIVE",
        "REVIEW_METHOD",
        "SEARCH_SOURCE",
        "SELECTION_FLOW",
        "ANALYSIS_AVAILABILITY",
        "PRISMA_CHECKLIST_DISPOSITION",
    },
    "MATERIAL_COVERAGE": None,
    "INTERPRETATION_CALIBRATION": {
        "NARRATIVE",
        "OUTCOME",
        "INTERPRETATION",
        "STUDY_OUTCOME",
        "SYNTHESIS",
        "RISK_OF_BIAS",
        "CERTAINTY",
    },
    "SOURCE_TRACEABILITY": None,
    "CROSS_SECTION_CONSISTENCY": {
        "NARRATIVE",
        "OUTCOME",
        "INTERPRETATION",
        "STUDY_OUTCOME",
        "SYNTHESIS",
        "CERTAINTY",
        "AUTHOR_RESPONSIBILITY",
    },
    "SCIENTIFIC_ORGANIZATION": set(),
}


_NO_VISIBLE_FACT_QUOTE = "NO_VISIBLE_FACT"


_RATING_ANCHOR_BY_RATING: dict[int, PublicationJudgeRatingAnchor] = {
    0: "0_CRITICAL_FAILURE",
    1: "1_MAJOR_FAILURE",
    2: "2_MATERIAL_WEAKNESS",
    3: "3_MINOR_LIMITATIONS",
    4: "4_NO_SUBSTANTIVE_ISSUE",
}


_SCORED_ASSESSMENT_PATTERN = re.compile(
    r"\ARATING: (?P<rating>[0-4])\n"
    r"RATING_ANCHOR: (?P<rating_anchor>[^\n]+)\n"
    r"CRITICAL_ISSUE: (?P<critical_issue>true|false)\n"
    r"EVIDENCE_BINDING: (?P<evidence_binding>[^\n]+)\n"
    r"IMPROVEMENT: (?P<improvement>[^\n]+)\n"
    r"ASSESSMENT: (?P<assessment>.+)\Z",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class _ScoredAssessment:
    rating: int
    rating_anchor: PublicationJudgeRatingAnchor
    critical_issue: bool
    improvement: str
    assessment: str


class PublicationManuscriptJudge:
    """Run a seven-dimensional, single-call diagnosis of a sealed V2 candidate outside its Run."""

    def __init__(
        self,
        *,
        candidate_root: Path,
        validation_root: Path,
        agent_runtime: Any,
        model_call_limiter: asyncio.Semaphore | None = None,
        score_mode: bool = False,
        structured_score: bool = False,
        compact_format: bool = False,
        array_score: bool = False,
    ) -> None:
        if array_score and (not structured_score or compact_format):
            raise ValueError("Array Judge format requires structured_score without compact_format")
        if structured_score and not score_mode:
            raise ValueError("Structured Judge output requires score_mode")
        if compact_format and not structured_score:
            raise ValueError("Compact Judge format requires structured_score")
        self.candidate_root = candidate_root.expanduser().resolve()
        self.validation_root = validation_root.expanduser().resolve()
        self.agent = SchemaAgentRunner(runtime=agent_runtime)
        self.model_call_limiter = model_call_limiter or asyncio.Semaphore(5)
        self.score_mode = score_mode
        self.structured_score = structured_score
        self.compact_format = compact_format
        self.array_score = array_score

    async def run(self) -> PublicationManuscriptJudgeResult:
        validation_id = str(uuid4())
        folder = RunFolder.create(self.validation_root / validation_id)
        try:
            candidate = self._load_candidate()
        except BaseException as error:
            failure_file = folder.write_immutable(
                "preflight-failure.json",
                canonical_json(
                    {
                        "schema_version": "manuscript-publication-judge-preflight-failure.v1",
                        "validation_id": validation_id,
                        "candidate_root": str(self.candidate_root),
                        "error_type": type(error).__name__,
                        "detail": str(error)[:4_000],
                    }
                ),
            )
            if not isinstance(error, Exception):
                raise
            raise PublicationJudgePreflightError(
                validation_id=validation_id,
                validation_root=folder.root,
                failure_path=folder.resolve(failure_file.path),
                cause=error,
            ) from error

        try:
            results = await asyncio.gather(
                *(
                    self._judge_one(
                        folder=folder,
                        validation_id=validation_id,
                        criterion=criterion,
                        candidate=candidate,
                    )
                    for criterion, _ in CRITERIA
                )
            )
        except BaseException as error:
            folder.write_immutable(
                "interruption.json",
                canonical_json(
                    {
                        "schema_version": "manuscript-publication-judge-interruption.v1",
                        "validation_id": validation_id,
                        "error_type": type(error).__name__,
                        "detail": str(error)[:4_000],
                    }
                ),
            )
            raise
        successful = tuple(
            item for item, _ in results if isinstance(item, PublicationJudgeCriterionOutput)
        )
        failed = tuple(
            criterion
            for (criterion, _), (item, _) in zip(CRITERIA, results, strict=True)
            if not isinstance(item, PublicationJudgeCriterionOutput)
        )
        score = None
        critical_issue_count = None
        if self.score_mode and not failed:
            ratings = {item.criterion_id: item.rating for item in successful}
            if any(value is None for value in ratings.values()):
                raise ValueError("Scored Publication Judge criterion lacks a rating")
            score = round(
                sum(weight * int(ratings[criterion]) / 4 for criterion, weight in CRITERIA),
                2,
            )
            critical_issue_count = sum(1 for item in successful if item.critical_issue)
        report = PublicationJudgeValidationReport(
            schema_version=(
                "manuscript-publication-judge-validation.v8"
                if self.score_mode
                else "manuscript-publication-judge-validation.v4"
            ),
            validation_id=validation_id,
            candidate_run_id=candidate.run_id,
            package_id=candidate.package.package_id,
            profile_id=candidate.delivery.profile_id,
            candidate_delivery_path=candidate.delivery_path,
            candidate_delivery_sha256=candidate.delivery_sha256,
            fact_package_sha256=candidate.fact_package_sha256,
            journal_profile_sha256=candidate.journal_profile_sha256,
            writer_projection_sha256=candidate.projection_sha256,
            claim_lineage_sha256=candidate.lineage_sha256,
            criterion_results=successful,
            failed_criteria=failed,
            criterion_artifacts=tuple(artifacts for _, artifacts in results),
            weighted_score=score,
            critical_issue_count=critical_issue_count,
            limitations=(
                "ROLE_ISOLATED_SAME_MODEL",
                "DIAGNOSTIC_NOT_HUMAN_ACCEPTANCE",
            ),
        )
        report_file = folder.write_immutable("report.json", canonical_json(report))
        return PublicationManuscriptJudgeResult(
            validation_id=validation_id,
            status="SUCCEEDED" if not failed else "COMPLETED_WITH_NA",
            validation_root=folder.root,
            report_path=folder.resolve(report_file.path),
            weighted_score=score,
            critical_issue_count=critical_issue_count,
        )

    async def _judge_one(
        self,
        *,
        folder: RunFolder,
        validation_id: str,
        criterion: JudgeCriterionId,
        candidate: _Candidate,
    ) -> tuple[
        PublicationJudgeCriterionOutput | Exception,
        PublicationJudgeCriterionArtifacts,
    ]:
        facts = self._facts_for_criterion(criterion, candidate.projection.facts)
        surfaces = tuple(candidate.surfaces[role] for role in _SURFACE_ROLES[criterion])
        surface_by_path = {surface.path: surface for surface in surfaces}
        fact_ids = {fact.fact_id for fact in facts}
        visible_claims = tuple(
            claim
            for claim in candidate.claims
            if any(
                path in surface_by_path and claim.claim_text in surface_by_path[path].content
                for path in claim.artifact_paths
            )
        )
        claims = (
            visible_claims
            if candidate.external_mode
            else tuple(
                claim
                for claim in visible_claims
                if (not fact_ids or set(claim.fact_ids) & fact_ids)
            )
        )
        if not claims:
            error = ValueError("No exact, criterion-visible claim evidence is available")
            return error, self._write_failure(folder, criterion, error, existing=())
        designated_claim = (
            claims[0]
            if candidate.external_mode
            else self._designated_claim(
                criterion,
                claims,
                {fact.fact_id: fact.kind for fact in candidate.projection.facts},
            )
        )
        visible_fact_by_id = {fact.fact_id: fact for fact in facts}
        designated_fact_id = (
            next(iter(visible_fact_by_id), candidate.projection.facts[0].fact_id)
            if candidate.external_mode
            else next(
                (fact_id for fact_id in designated_claim.fact_ids if fact_id in visible_fact_by_id),
                designated_claim.fact_ids[0],
            )
        )

        semantic = PublicationJudgeCriterionInput(
            criterion_id=criterion,
            package_id=candidate.package.package_id,
            profile_id=candidate.delivery.profile_id,
            candidate_delivery_sha256=candidate.delivery_sha256,
            surfaces=surfaces,
            facts=facts,
            headline_decisions=(
                candidate.projection.headline_decisions
                if criterion
                in {
                    "FACTUAL_FIDELITY",
                    "MATERIAL_COVERAGE",
                    "INTERPRETATION_CALIBRATION",
                    "CROSS_SECTION_CONSISTENCY",
                }
                else ()
            ),
            claim_lineage=claims,
            designated_claim_id=designated_claim.claim_id,
            designated_fact_id=designated_fact_id,
            rating_anchors=RATING_ANCHORS,
        )
        template = (
            publication_judge_template if candidate.external_mode else native_judge_template
        )(
            criterion,
            score_mode=self.score_mode,
            structured_score=self.structured_score,
            compact_format=self.compact_format,
            array_score=self.array_score,
        )
        stage_payload = self.agent.stage_payload(
            template,
            semantic.model_dump(mode="json"),
            program_binding={
                "validation_id": validation_id,
                "candidate_run_id": candidate.run_id,
                "candidate_delivery_sha256": candidate.delivery_sha256,
                "fact_package_sha256": candidate.fact_package_sha256,
                "writer_projection_sha256": candidate.projection_sha256,
                "claim_lineage_sha256": candidate.lineage_sha256,
                "criterion_id": criterion,
                "model": "deepseek-v4-flash",
                "temperature": 0,
                "retry_policy": "NONE",
                "tools": [],
                "candidate_origin": (candidate.candidate_origin),
            },
        )
        criterion_root = f"criteria/{criterion.lower()}"
        input_file = folder.write_immutable(
            f"{criterion_root}/input.json", canonical_json(stage_payload)
        )
        input_artifact = PublicationJudgeArtifactFile(
            role="INPUT",
            path=input_file.path,
            sha256=input_file.sha256,
        )
        try:
            async with self.model_call_limiter:
                response = await self.agent.invoke_stage(template, stage_payload)
            payload = response.payload
            if self.array_score:
                payload = dict(
                    zip(PublicationJudgeStructuredScore.model_fields, payload["score"], strict=True)
                )
            agent_result = (
                PublicationJudgeStructuredScore
                if self.structured_score
                else PublicationJudgeCriterionAgentOutput
            ).model_validate(payload)
            rating = None
            rating_anchor = None
            critical_issue = None
            manuscript_quote = None
            fact_quote = None
            improvement = None
            evidence_binding = None
            assessment = agent_result.assessment
            if self.score_mode:
                scored = (
                    agent_result
                    if self.structured_score
                    else self._parse_scored_assessment(agent_result.assessment)
                )
                rating = scored.rating
                rating_anchor = scored.rating_anchor
                critical_issue = scored.critical_issue
                manuscript_quote = designated_claim.claim_text
                fact_quote = (
                    visible_fact_by_id[designated_fact_id].statement_en
                    if designated_fact_id in visible_fact_by_id
                    else _NO_VISIBLE_FACT_QUOTE
                )
                improvement = scored.improvement
                evidence_binding = (
                    "PROGRAM_DESIGNATED_CONTEXT_FACT_NO_CLAIM_BINDING"
                    if candidate.external_mode
                    else "PROGRAM_DESIGNATED_CLAIM_AND_FACT"
                )
                assessment = scored.assessment
            artifact_path = next(
                (
                    path
                    for path in designated_claim.artifact_paths
                    if path in surface_by_path
                    and designated_claim.claim_text in surface_by_path[path].content
                ),
                None,
            )
            if artifact_path is None:
                raise ValueError("Publication Judge claim is not exact in a bound surface")
            result = PublicationJudgeCriterionOutput(
                schema_version=(
                    "manuscript-publication-judge-criterion.v7"
                    if self.score_mode
                    else "manuscript-publication-judge-criterion.v3"
                ),
                criterion_id=criterion,
                manuscript_evidence=(
                    PublicationJudgeEvidence(
                        claim_id=designated_claim.claim_id,
                        artifact_path=artifact_path,
                        quote=designated_claim.claim_text,
                    ),
                ),
                fact_evidence=(
                    FactJudgeEvidence(
                        fact_id=designated_fact_id,
                        relation=(
                            "Common Judge context fact; it is not bound to the external claim."
                            if candidate.external_mode
                            else "Program-designated criterion evidence; see assessment."
                        ),
                    ),
                ),
                rating=rating,
                rating_anchor=rating_anchor,
                critical_issue=critical_issue,
                manuscript_quote=manuscript_quote,
                fact_quote=fact_quote,
                improvement=improvement,
                evidence_binding=evidence_binding,
                assessment=assessment,
            )
            result_file = folder.write_immutable(
                f"{criterion_root}/result.json", canonical_json(result)
            )
            metrics_file = folder.write_immutable(
                f"{criterion_root}/metrics.json",
                canonical_json(
                    {
                        "schema_version": "manuscript-publication-judge-metrics.v1",
                        "criterion_id": criterion,
                        "metrics": response.metrics,
                    }
                ),
            )
            raw_file = folder.write_immutable(
                f"{criterion_root}/raw-output.json",
                json.dumps(
                    response.raw_model_output,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            artifacts = PublicationJudgeCriterionArtifacts(
                criterion_id=criterion,
                status="SUCCEEDED",
                files=(
                    input_artifact,
                    PublicationJudgeArtifactFile(
                        role="RAW_OUTPUT", path=raw_file.path, sha256=raw_file.sha256
                    ),
                    PublicationJudgeArtifactFile(
                        role="RESULT", path=result_file.path, sha256=result_file.sha256
                    ),
                    PublicationJudgeArtifactFile(
                        role="METRICS", path=metrics_file.path, sha256=metrics_file.sha256
                    ),
                ),
            )
            return result, artifacts
        except Exception as error:
            if "response" in locals() and not isinstance(error, SchemaAgentOutputError):
                error = SchemaAgentOutputError(
                    str(error),
                    raw_model_output=response.raw_model_output,
                    metrics=response.metrics,
                )
            return error, self._write_failure(
                folder,
                criterion,
                error,
                existing=(input_artifact,),
            )

    @staticmethod
    def _facts_for_criterion(
        criterion: JudgeCriterionId,
        facts: tuple[PublicationWriterFact, ...],
    ) -> tuple[PublicationWriterFact, ...]:
        selected = _FACT_KINDS[criterion]
        return facts if selected is None else tuple(fact for fact in facts if fact.kind in selected)

    @staticmethod
    def _designated_claim(
        criterion: JudgeCriterionId,
        claims: tuple[PublicationJudgeClaim, ...],
        fact_kinds: dict[str, str],
    ) -> PublicationJudgeClaim:
        priorities = {
            "FACTUAL_FIDELITY": ("SYNTHESIS", "STUDY_OUTCOME", "NARRATIVE"),
            "PROTOCOL_AND_CONDUCT_FIDELITY": (
                "NARRATIVE",
                "REVIEW_METHOD",
                "SEARCH_SOURCE",
                "SELECTION_FLOW",
            ),
            "MATERIAL_COVERAGE": ("SYNTHESIS", "STUDY_OUTCOME", "OUTCOME", "SELECTION_FLOW"),
            "INTERPRETATION_CALIBRATION": (
                "SYNTHESIS",
                "INTERPRETATION",
                "OUTCOME",
                "CERTAINTY",
                "NARRATIVE",
            ),
            "SOURCE_TRACEABILITY": ("STUDY_OUTCOME", "SYNTHESIS", "SEARCH_SOURCE"),
            "CROSS_SECTION_CONSISTENCY": (
                "SYNTHESIS",
                "INTERPRETATION",
                "OUTCOME",
                "CERTAINTY",
                "NARRATIVE",
            ),
            "SCIENTIFIC_ORGANIZATION": ("NARRATIVE",),
        }[criterion]
        for kind in priorities:
            for claim in claims:
                if fact_kinds.get(claim.fact_ids[0]) == kind:
                    return claim
        return claims[0]

    @staticmethod
    def _parse_scored_assessment(value: str) -> _ScoredAssessment:
        match = _SCORED_ASSESSMENT_PATTERN.fullmatch(value.strip())
        if match is None:
            raise ValueError(
                "Scored Publication Judge output must use exact RATING, RATING_ANCHOR, "
                "CRITICAL_ISSUE, EVIDENCE_BINDING, IMPROVEMENT, and ASSESSMENT lines"
            )
        rating = int(match.group("rating"))
        rating_anchor = _RATING_ANCHOR_BY_RATING[rating]
        raw_rating_anchor = match.group("rating_anchor")
        numeric_anchor = re.match(r"(?P<rating>[0-4])(?:_|:)", raw_rating_anchor)
        if numeric_anchor is not None and int(numeric_anchor.group("rating")) != rating:
            raise ValueError("Scored Publication Judge rating anchor contradicts rating")
        named_anchors = tuple(_RATING_ANCHOR_BY_RATING.values())
        mismatched_name = any(
            anchor != rating_anchor
            and (anchor in raw_rating_anchor or anchor[2:] in raw_rating_anchor)
            for anchor in named_anchors
        )
        if mismatched_name:
            raise ValueError("Scored Publication Judge rating anchor contradicts rating")
        assessment = match.group("assessment").strip()
        evidence_binding = match.group("evidence_binding").strip()
        improvement = match.group("improvement").strip()
        if not assessment or not evidence_binding or not improvement:
            raise ValueError("Scored Publication Judge anchored fields must be non-empty")
        if rating == 0 and match.group("critical_issue") != "true":
            raise ValueError("Scored Publication Judge rating 0 must be critical")
        for label, text, limit in (
            ("assessment", assessment, 4_000),
            ("improvement", improvement, 2_000),
        ):
            if len(text) > limit:
                raise ValueError(f"Scored Publication Judge {label} exceeds {limit} characters")
        return _ScoredAssessment(
            rating=rating,
            rating_anchor=rating_anchor,
            critical_issue=match.group("critical_issue") == "true",
            improvement=improvement,
            assessment=assessment,
        )

    @staticmethod
    def _write_failure(
        folder: RunFolder,
        criterion: JudgeCriterionId,
        error: Exception,
        *,
        existing: tuple[PublicationJudgeArtifactFile, ...],
    ) -> PublicationJudgeCriterionArtifacts:
        root = f"criteria/{criterion.lower()}"
        failure_file = folder.write_immutable(
            f"{root}/failure.json",
            canonical_json(
                {
                    "schema_version": "manuscript-publication-judge-failure.v1",
                    "criterion_id": criterion,
                    "status": "N/A",
                    "error_type": type(error).__name__,
                    "detail": str(error)[:4_000],
                    "raw_model_output": getattr(error, "raw_model_output", None),
                    "metrics": getattr(error, "metrics", None),
                }
            ),
        )
        metrics_file = folder.write_immutable(
            f"{root}/metrics.json",
            canonical_json(
                {
                    "schema_version": "manuscript-publication-judge-metrics.v1",
                    "criterion_id": criterion,
                    "metrics": getattr(error, "metrics", None),
                }
            ),
        )
        files = [
            *existing,
            PublicationJudgeArtifactFile(
                role="FAILURE", path=failure_file.path, sha256=failure_file.sha256
            ),
            PublicationJudgeArtifactFile(
                role="METRICS", path=metrics_file.path, sha256=metrics_file.sha256
            ),
        ]
        raw_output = getattr(error, "raw_model_output", None)
        if raw_output is not None:
            raw_file = folder.write_immutable(
                f"{root}/raw-output.json",
                json.dumps(
                    raw_output,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            files.append(
                PublicationJudgeArtifactFile(
                    role="RAW_OUTPUT", path=raw_file.path, sha256=raw_file.sha256
                )
            )
        return PublicationJudgeCriterionArtifacts(
            criterion_id=criterion,
            status="FAILED",
            files=tuple(files),
        )

    def _load_candidate(self) -> _Candidate:
        folder = RunFolder(self.candidate_root)
        manifest = RunManifest(folder.manifest_path)
        with manifest.connect() as connection:
            runs = connection.execute("SELECT * FROM runs").fetchall()
            stages = connection.execute(
                "SELECT * FROM stage_results ORDER BY started_at, result_id"
            ).fetchall()
        if len(runs) != 1:
            raise ValueError("Candidate manifest must contain exactly one Run")
        run = runs[0]
        if run["specialist"] != "manuscript":
            raise ValueError("Candidate is not a Manuscript Run")
        if run["status"] not in {"SUCCEEDED", "COMPLETED_WITH_ISSUES"}:
            raise ValueError("Candidate Manuscript Run is not sealed")
        run_input = json.loads(folder.read_verified(run["input_path"], run["input_sha256"]))
        if not stages or any(stage["status"] == "RUNNING" for stage in stages):
            raise ValueError("Candidate contains an incomplete Stage Result")
        for stage in stages:
            folder.read_verified(stage["input_path"], stage["input_sha256"])
            if stage["output_path"] is None or stage["output_sha256"] is None:
                raise ValueError("Candidate Stage Result lacks a canonical output")
            folder.read_verified(stage["output_path"], stage["output_sha256"])

        final_bytes = folder.read_verified(run["final_output_path"], run["final_output_sha256"])
        final = ManuscriptRunResultV2.model_validate_json(final_bytes)
        if final.run_id != run["run_id"] or final.status != run["status"]:
            raise ValueError("Candidate final result contradicts the Run manifest")
        if final.delivery_path is None or final.delivery_sha256 is None:
            raise ValueError("Candidate final result lacks a sealed delivery")
        delivery_bytes = folder.read_verified(final.delivery_path, final.delivery_sha256)
        delivery = ManuscriptDeliveryV2.model_validate_json(delivery_bytes)
        if (
            delivery.run_id != run["run_id"]
            or delivery.package_id != final.package_id
            or delivery.profile_id != final.profile_id
        ):
            raise ValueError("Candidate delivery contradicts the final result")
        files_by_role = {item.role: item for item in delivery.files}
        if len(files_by_role) != len(delivery.files):
            raise ValueError("Candidate delivery contains duplicate roles")
        if len({item.path for item in delivery.files}) != len(delivery.files):
            raise ValueError("Candidate delivery contains duplicate paths")
        delivery_content = {
            item.role: folder.read_verified(item.path, item.sha256) for item in delivery.files
        }

        import_rows = [stage for stage in stages if stage["stage"] == "publication_input_import"]
        compile_rows = [
            stage for stage in stages if stage["stage"] == "publication_profile_compile"
        ]
        if len(import_rows) != 1 or len(compile_rows) != 1:
            raise ValueError("Candidate lacks an exact V2 import or profile-compile Stage")
        import_output = json.loads(
            folder.read_verified(import_rows[0]["output_path"], import_rows[0]["output_sha256"])
        )
        copied_input_bindings: dict[str, str] = {}
        for copied in import_output.get("copied_inputs", []):
            if not isinstance(copied, dict) or not isinstance(copied.get("path"), str):
                raise ValueError("Candidate input-import audit is malformed")
            if not isinstance(copied.get("sha256"), str):
                raise ValueError("Candidate input-import hash is malformed")
            if copied["path"] in copied_input_bindings:
                raise ValueError("Candidate input-import contains duplicate paths")
            copied_input_bindings[copied["path"]] = copied["sha256"]
            folder.read_verified(copied["path"], copied["sha256"])
        compile_output = json.loads(
            folder.read_verified(compile_rows[0]["output_path"], compile_rows[0]["output_sha256"])
        )
        package_bytes = folder.read_verified(
            "inputs/fact-package.json", import_output["package_sha256"]
        )
        package = ManuscriptFactPackageV2.model_validate_json(package_bytes)
        projection_bytes = folder.read_verified(
            compile_output["writer_projection_path"],
            compile_output["writer_projection_sha256"],
        )
        projection = PublicationWriterProjection.model_validate_json(projection_bytes)
        if (
            package.package_id != delivery.package_id
            or projection.package_id != package.package_id
            or projection.profile_id != delivery.profile_id
        ):
            raise ValueError("Candidate package, projection, and delivery identities differ")
        expected_projection = PublicationWriterProjection(
            schema_version="manuscript-writer-projection.v2",
            package_id=package.package_id,
            profile_id=delivery.profile_id,
            facts=tuple(
                PublicationWriterFact(
                    fact_id=fact.fact_id,
                    kind=fact.kind,
                    status=fact.status,
                    statement_en=fact.statement_en,
                    required_for=fact.required_for,
                    allowed_placements=fact.allowed_placements,
                    citation_keys=fact.citation_keys,
                    issue_refs=fact.issue_refs,
                )
                for fact in package.facts
            ),
            headline_decisions=tuple(
                PublicationHeadlineDecision(
                    analysis_id=item.analysis_id,
                    eligibility=item.eligibility,
                    reason=item.reason,
                )
                for item in decide_headlines(package)
            ),
        )
        if projection != expected_projection:
            raise ValueError("Candidate writer projection is not the exact Fact Package projection")

        external_mode = False
        candidate_origin = "METARIGOR_RUN"
        external_metadata: dict[str, Any] | None = None
        external_metadata_path = "inputs/external-candidate.json"
        external_metadata_file = folder.resolve(external_metadata_path)
        external_metadata_binding = import_output.get("external_metadata_sha256")
        if external_metadata_file.is_file() and not external_metadata_binding:
            raise ValueError("External candidate metadata is not bound by input-import Stage")
        if external_metadata_binding and not external_metadata_file.is_file():
            raise ValueError("Bound external candidate metadata is missing")
        if external_metadata_binding != copied_input_bindings.get(external_metadata_path):
            raise ValueError("External candidate metadata hash is not sealed by input-import Stage")
        if external_metadata_file.is_file():
            external_metadata = json.loads(
                folder.read_verified(external_metadata_path, external_metadata_binding)
            )
            normalization_mode = external_metadata.get("normalization_mode")
            if (
                external_metadata.get("schema_version") != "manuscript-external-candidate.v1"
                or external_metadata.get("run_id") != run["run_id"]
                or normalization_mode
                not in {
                    "BLIND_CONTENT_ONLY",
                    "DIRECT_ABLATION_GENERATION",
                    "SEALED_CLI_CONTENT_ONLY",
                }
                or external_metadata.get("semantic_enrichment") is not False
                or external_metadata.get("claim_lineage_mode")
                != "STRUCTURAL_RAW_SURFACE_WRAPPER_NO_FACT_ASSERTION"
                or external_metadata.get("source_anchor_binding") != "NONE_ADDED_BY_NORMALIZER"
            ):
                raise ValueError(
                    "External candidate metadata is not a supported sealed-content contract"
                )
            if normalization_mode == "DIRECT_ABLATION_GENERATION" and (
                external_metadata.get("generation_condition")
                not in {
                    "NO_TASKCONTRACT_DECOMPOSITION",
                    "NO_PROGRAM_MODEL_RESPONSIBILITY_SPLIT",
                }
                or external_metadata.get("generation_run_id") != run["run_id"]
            ):
                raise ValueError("Direct ablation candidate identity is malformed")
            if normalization_mode == "DIRECT_ABLATION_GENERATION":
                condition = external_metadata["generation_condition"]
                if (
                    run_input.get("condition") != condition
                    or run_input.get("run_id") != run["run_id"]
                    or run_input.get("case_id") != external_metadata.get("case_id")
                ):
                    raise ValueError("Direct ablation metadata contradicts the sealed Run input")
                observed_model_stages = {
                    stage["stage"]
                    for stage in stages
                    if stage["stage"] == "whole_manuscript"
                    or stage["stage"].startswith("manuscript_")
                }
                expected_model_stages = (
                    {"whole_manuscript"}
                    if condition == "NO_TASKCONTRACT_DECOMPOSITION"
                    else {
                        "manuscript_plan",
                        "manuscript_title_abstract",
                        "manuscript_introduction",
                        "manuscript_methods",
                        "manuscript_results",
                        "manuscript_discussion",
                    }
                )
                if observed_model_stages != expected_model_stages:
                    raise ValueError("Direct ablation model Stage topology differs")
                candidate_origin = "DIRECT_ABLATION_GENERATION"
            elif normalization_mode == "SEALED_CLI_CONTENT_ONLY":
                if package.fact_scope != "REFERENCE_UPSTREAM_PROXY":
                    raise ValueError("Sealed CLI mode requires a clean proxy package")
                candidate_origin = "SEALED_CLI_GENERATION"
            else:
                candidate_origin = "EXTERNAL_BLIND_CONTENT"
            external_mode = True
            raw_artifacts = external_metadata.get("raw_artifacts")
            if not isinstance(raw_artifacts, list) or not raw_artifacts:
                raise ValueError("External candidate metadata lacks raw-content bindings")
            for raw_artifact in raw_artifacts:
                if not isinstance(raw_artifact, dict):
                    raise ValueError("External raw-content binding is malformed")
                normalized_path = raw_artifact.get("normalized_path")
                normalized_sha256 = raw_artifact.get("normalized_sha256")
                if (
                    not isinstance(normalized_path, str)
                    or not normalized_path.startswith(
                        "raw-output/"
                        if normalization_mode == "DIRECT_ABLATION_GENERATION"
                        else "inputs/external-content/"
                    )
                    or not isinstance(normalized_sha256, str)
                ):
                    raise ValueError("External raw-content binding path is invalid")
                folder.read_verified(normalized_path, normalized_sha256)

        if external_mode and package.fact_scope == "REFERENCE_UPSTREAM_PROXY":
            from .ms_ablation import validate_proxy_ablation_inputs

            if run_input.get("writer_mode") == "LLM" or candidate_origin not in {
                "DIRECT_ABLATION_GENERATION",
                "SEALED_CLI_GENERATION",
            }:
                raise ValueError(
                    "Proxy external candidate must be a direct ablation or sealed CLI, "
                    "not a native writer"
                )
            validator = validate_proxy_ablation_inputs
            if candidate_origin == "SEALED_CLI_GENERATION":
                from .ms_cli_proxy import validate_proxy_cli_inputs

                validator = validate_proxy_cli_inputs
            validator(
                folder,
                stages,
                run_input,
                package,
                projection,
                copied_input_bindings,
                delivery_content,
            )
        elif (
            package.fact_scope == "REFERENCE_UPSTREAM_PROXY"
            or run_input.get("writer_mode") == "LLM"
        ):
            from .publication_writing import validate_candidate_writing

            validate_candidate_writing(
                folder,
                stages,
                run_input,
                package,
                delivery,
                delivery_content,
                copied_input_bindings,
            )

        required_roles = set().union(*_SURFACE_ROLES.values(), {"CLAIM_LINEAGE"})
        if required_roles - files_by_role.keys():
            raise ValueError("Candidate delivery lacks a required Judge surface")
        surfaces = {
            role: PublicationJudgeSurface(
                role=role,
                path=files_by_role[role].path,
                sha256=files_by_role[role].sha256,
                content=delivery_content[role].decode("utf-8"),
            )
            for role in set().union(*_SURFACE_ROLES.values())
        }
        lineage_file = files_by_role["CLAIM_LINEAGE"]
        raw_claims = tuple(
            PublicationClaimLineageV2.model_validate_json(line)
            for line in delivery_content["CLAIM_LINEAGE"].splitlines()
            if line.strip()
        )
        if not raw_claims or len({claim.claim_id for claim in raw_claims}) != len(raw_claims):
            raise ValueError("Candidate claim lineage is empty or has duplicate ids")
        facts_by_id = {fact.fact_id: fact for fact in package.facts}
        paths = {item.path for item in delivery.files}
        claims = []
        for claim in raw_claims:
            if claim.profile_id != delivery.profile_id:
                raise ValueError("Candidate Claim has invalid Profile or Fact cardinality")
            if external_mode:
                if claim.fact_ids:
                    raise ValueError("External Claim wrapper must not cite a Publication Fact")
                fact = None
            else:
                if len(claim.fact_ids) != 1:
                    raise ValueError("Candidate Claim has invalid Profile or Fact cardinality")
                fact = facts_by_id.get(claim.fact_ids[0])
                if fact is None:
                    raise ValueError("Candidate Claim cites an unknown Publication Fact")
            if set(claim.artifact_paths) - paths:
                raise ValueError("Candidate Claim cites an artifact outside the delivery")
            if not any(
                claim.claim_text in delivery_content[role].decode("utf-8")
                for role in files_by_role
                if files_by_role[role].path in claim.artifact_paths
            ):
                raise ValueError("Candidate Claim quote is absent from its bound artifact")
            if external_mode:
                if (
                    claim.source_anchors
                    or claim.terminal_source_bindings
                    or claim.upstream_artifact_refs
                    or claim.status != "NOT_REPORTED"
                ):
                    raise ValueError(
                        "External Claim wrapper must not add provenance or reported status"
                    )
            elif (
                claim.allowed_placements != fact.allowed_placements
                or claim.source_anchors != fact.source_anchors
                or claim.terminal_source_bindings != fact.terminal_source_bindings
                or claim.upstream_artifact_refs != fact.upstream_artifact_refs
                or claim.status != fact.status
            ):
                raise ValueError("Candidate Claim provenance contradicts its Publication Fact")
            claims.append(
                PublicationJudgeClaim(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    fact_ids=claim.fact_ids,
                    artifact_paths=claim.artifact_paths,
                    status=claim.status,
                    source_anchor_ids=tuple(item.anchor_id for item in claim.source_anchors),
                    terminal_source_binding_count=len(claim.terminal_source_bindings),
                )
            )
        return _Candidate(
            run_id=run["run_id"],
            package=package,
            projection=projection,
            delivery=delivery,
            delivery_path=final.delivery_path,
            delivery_sha256=final.delivery_sha256,
            fact_package_sha256=hashlib.sha256(package_bytes).hexdigest(),
            journal_profile_sha256=delivery.profile_sha256,
            projection_sha256=hashlib.sha256(projection_bytes).hexdigest(),
            lineage_sha256=lineage_file.sha256,
            claims=tuple(claims),
            surfaces=surfaces,
            external_mode=external_mode,
            candidate_origin=candidate_origin,
        )

