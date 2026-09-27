from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator
from .models import FactJudgeEvidence, Identifier, JudgeCriterionId, ManuscriptModel, Sha256, SourceAnchor
from .publication_models import PublicationPlacement, PublicationStatus, TerminalSourceBinding
PublicationJudgeRatingAnchor = Literal[
    "0_CRITICAL_FAILURE",
    "1_MAJOR_FAILURE",
    "2_MATERIAL_WEAKNESS",
    "3_MINOR_LIMITATIONS",
    "4_NO_SUBSTANTIVE_ISSUE",
]


_RATING_ANCHOR_BY_RATING = {
    0: "0_CRITICAL_FAILURE",
    1: "1_MAJOR_FAILURE",
    2: "2_MATERIAL_WEAKNESS",
    3: "3_MINOR_LIMITATIONS",
    4: "4_NO_SUBSTANTIVE_ISSUE",
}


class PublicationWriterFact(ManuscriptModel):
    fact_id: Identifier
    kind: Literal[
        "NARRATIVE",
        "OUTCOME",
        "REVIEW_METHOD",
        "INTERPRETATION",
        "SEARCH_SOURCE",
        "SELECTION_FLOW",
        "STUDY",
        "STUDY_OUTCOME",
        "SYNTHESIS",
        "ANALYSIS_AVAILABILITY",
        "RISK_OF_BIAS",
        "CERTAINTY",
        "AUTHOR_RESPONSIBILITY",
        "PRISMA_CHECKLIST_DISPOSITION",
    ]
    status: PublicationStatus
    statement_en: str = Field(min_length=1, max_length=8_000)
    required_for: tuple[Identifier, ...] = Field(min_length=1, max_length=100)
    allowed_placements: tuple[PublicationPlacement, ...] = Field(min_length=1, max_length=5)
    citation_keys: tuple[Identifier, ...] = Field(default=(), max_length=100)
    issue_refs: tuple[Identifier, ...] = Field(default=(), max_length=30)


class PublicationHeadlineDecision(ManuscriptModel):
    analysis_id: Identifier
    eligibility: Literal["HEADLINE_ELIGIBLE", "AUDIT_ONLY"]
    reason: str = Field(min_length=1, max_length=4_000)


class PublicationWriterProjection(ManuscriptModel):
    schema_version: Literal["manuscript-writer-projection.v2"]
    package_id: Identifier
    profile_id: Identifier
    facts: tuple[PublicationWriterFact, ...] = Field(min_length=1, max_length=20_000)
    headline_decisions: tuple[PublicationHeadlineDecision, ...] = Field(max_length=1_000)

    @model_validator(mode="after")
    def ids_are_unique(self) -> PublicationWriterProjection:
        fact_ids = [fact.fact_id for fact in self.facts]
        decision_ids = [item.analysis_id for item in self.headline_decisions]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("Publication writer projection contains duplicate Fact ids")
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("Publication writer projection contains duplicate analysis ids")
        return self


class PublicationJudgeClaim(ManuscriptModel):
    claim_id: Identifier
    claim_text: str = Field(min_length=1, max_length=8_000)
    fact_ids: tuple[Identifier, ...] = Field(default=(), max_length=100)
    artifact_paths: tuple[str, ...] = Field(min_length=1, max_length=20)
    status: PublicationStatus
    source_anchor_ids: tuple[Identifier, ...] = Field(default=(), max_length=100)
    terminal_source_binding_count: int = Field(ge=0, le=100)


class PublicationClaimLineageV2(ManuscriptModel):
    schema_version: Literal["manuscript-claim-lineage.v2"]
    claim_id: Identifier
    profile_id: Identifier
    fact_ids: tuple[Identifier, ...] = Field(default=(), max_length=100)
    claim_text: str = Field(min_length=1, max_length=8_000)
    artifact_paths: tuple[str, ...] = Field(min_length=1, max_length=20)
    allowed_placements: tuple[PublicationPlacement, ...] = Field(min_length=1, max_length=5)
    source_anchors: tuple[SourceAnchor, ...] = Field(default=(), max_length=100)
    terminal_source_bindings: tuple[TerminalSourceBinding, ...] = Field(default=(), max_length=100)
    upstream_artifact_refs: tuple[Identifier, ...] = Field(default=(), max_length=30)
    status: PublicationStatus


class PublicationJudgeSurface(ManuscriptModel):
    role: str = Field(min_length=1, max_length=100)
    path: str = Field(min_length=1, max_length=1_000)
    sha256: Sha256
    content: str = Field(min_length=1, max_length=2_000_000)


class PublicationJudgeCriterionInput(ManuscriptModel):
    criterion_id: JudgeCriterionId
    package_id: Identifier
    profile_id: Identifier
    candidate_delivery_sha256: Sha256
    surfaces: tuple[PublicationJudgeSurface, ...] = Field(min_length=1, max_length=10)
    facts: tuple[PublicationWriterFact, ...] = Field(max_length=20_000)
    headline_decisions: tuple[PublicationHeadlineDecision, ...] = Field(max_length=1_000)
    claim_lineage: tuple[PublicationJudgeClaim, ...] = Field(min_length=1, max_length=20_000)
    designated_claim_id: Identifier
    designated_fact_id: Identifier
    rating_anchors: tuple[str, ...] = Field(min_length=5, max_length=5)


class PublicationJudgeEvidence(ManuscriptModel):
    claim_id: Identifier
    artifact_path: str = Field(min_length=1, max_length=1_000)
    quote: str = Field(min_length=1, max_length=8_000)


class PublicationJudgeCriterionAgentOutput(ManuscriptModel):
    assessment: str = Field(min_length=1, max_length=4_000)


class PublicationJudgeStructuredScore(ManuscriptModel):
    """同一七维 Judge 的 JSON 输出；程序仍拥有证据和最终分数聚合。"""

    rating: int = Field(ge=0, le=4, strict=True, json_schema_extra={"enum": [0, 1, 2, 3, 4]})
    rating_anchor: PublicationJudgeRatingAnchor
    critical_issue: bool = Field(strict=True)
    evidence_binding: str = Field(min_length=1, max_length=2_000)
    improvement: str = Field(min_length=1, max_length=2_000)
    assessment: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def score_is_consistent(self) -> PublicationJudgeStructuredScore:
        if self.rating_anchor != _RATING_ANCHOR_BY_RATING[self.rating]:
            raise ValueError("Scored Publication Judge rating anchor contradicts rating")
        if self.rating == 0 and not self.critical_issue:
            raise ValueError("Scored Publication Judge rating 0 must be critical")
        if not all(v.strip() for v in (self.evidence_binding, self.improvement, self.assessment)):
            raise ValueError("Scored Publication Judge anchored fields must be non-empty")
        # 保持旧六行响应的总字符预算；这里只计长，不再按位置解析模型输出。
        equivalent_text = "\n".join(
            (
                f"RATING: {self.rating}",
                f"RATING_ANCHOR: {self.rating_anchor}",
                f"CRITICAL_ISSUE: {str(self.critical_issue).lower()}",
                f"EVIDENCE_BINDING: {self.evidence_binding}",
                f"IMPROVEMENT: {self.improvement}",
                f"ASSESSMENT: {self.assessment}",
            )
        )
        if len(equivalent_text) > 4_000:
            raise ValueError("Scored Publication Judge response exceeds 4000 characters")
        return self


class PublicationJudgeCriterionOutput(ManuscriptModel):
    schema_version: Literal[
        "manuscript-publication-judge-criterion.v3",
        "manuscript-publication-judge-criterion.v4",
        "manuscript-publication-judge-criterion.v5",
        "manuscript-publication-judge-criterion.v6",
        "manuscript-publication-judge-criterion.v7",
    ]
    criterion_id: JudgeCriterionId
    manuscript_evidence: tuple[PublicationJudgeEvidence, ...] = Field(min_length=1, max_length=1)
    fact_evidence: tuple[FactJudgeEvidence, ...] = Field(min_length=1, max_length=1)
    rating: int | None = Field(default=None, ge=0, le=4)
    rating_anchor: PublicationJudgeRatingAnchor | None = None
    critical_issue: bool | None = None
    manuscript_quote: str | None = Field(default=None, min_length=1, max_length=8_000)
    fact_quote: str | None = Field(default=None, min_length=1, max_length=8_000)
    improvement: str | None = Field(default=None, min_length=1, max_length=2_000)
    evidence_binding: (
        Literal[
            "PROGRAM_DESIGNATED_CLAIM_AND_FACT",
            "PROGRAM_DESIGNATED_CONTEXT_FACT_NO_CLAIM_BINDING",
        ]
        | None
    ) = None
    assessment: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def score_fields_match_schema_version(self) -> PublicationJudgeCriterionOutput:
        if self.schema_version == "manuscript-publication-judge-criterion.v3":
            if (
                self.rating is not None
                or self.rating_anchor is not None
                or self.critical_issue is not None
                or self.manuscript_quote is not None
                or self.fact_quote is not None
                or self.improvement is not None
                or self.evidence_binding is not None
            ):
                raise ValueError("Publication Judge criterion v3 is qualitative only")
        elif self.rating is None or self.critical_issue is None:
            raise ValueError(
                "Publication Judge scored criterion must carry rating and critical flag"
            )
        elif self.schema_version == "manuscript-publication-judge-criterion.v4":
            if (
                self.rating_anchor is not None
                or self.manuscript_quote is not None
                or self.fact_quote is not None
                or self.improvement is not None
                or self.evidence_binding is not None
            ):
                raise ValueError("Publication Judge criterion v4 cannot carry anchor fields")
        elif self.schema_version == "manuscript-publication-judge-criterion.v5":
            if (
                self.rating_anchor is None
                or self.manuscript_quote is None
                or self.fact_quote is None
                or self.improvement is None
            ):
                raise ValueError("Publication Judge criterion v5 must carry anchored evidence")
            if self.evidence_binding is not None:
                raise ValueError("Publication Judge criterion v5 cannot carry evidence binding")
            if self.rating_anchor != _RATING_ANCHOR_BY_RATING[self.rating]:
                raise ValueError("Publication Judge rating anchor contradicts rating")
            if self.rating == 0 and not self.critical_issue:
                raise ValueError("Publication Judge rating 0 must be critical")
        elif self.schema_version == "manuscript-publication-judge-criterion.v6":
            if (
                self.rating_anchor is None
                or self.manuscript_quote is None
                or self.fact_quote is None
                or self.improvement is None
                or self.evidence_binding is None
            ):
                raise ValueError("Publication Judge criterion v6 must carry program-bound evidence")
            if self.rating_anchor != _RATING_ANCHOR_BY_RATING[self.rating]:
                raise ValueError("Publication Judge rating anchor contradicts rating")
            if self.rating == 0 and not self.critical_issue:
                raise ValueError("Publication Judge rating 0 must be critical")
        else:
            if (
                self.rating_anchor is None
                or self.manuscript_quote is None
                or self.fact_quote is None
                or self.improvement is None
                or self.evidence_binding is None
            ):
                raise ValueError("Publication Judge criterion v7 must carry program-bound evidence")
            if self.rating_anchor != _RATING_ANCHOR_BY_RATING[self.rating]:
                raise ValueError("Publication Judge rating anchor contradicts rating")
            if self.rating == 0 and not self.critical_issue:
                raise ValueError("Publication Judge rating 0 must be critical")
        return self


class PublicationJudgeArtifactFile(ManuscriptModel):
    role: Literal["INPUT", "RAW_OUTPUT", "RESULT", "METRICS", "FAILURE"]
    path: str = Field(min_length=1, max_length=1_000)
    sha256: Sha256


class PublicationJudgeCriterionArtifacts(ManuscriptModel):
    criterion_id: JudgeCriterionId
    status: Literal["SUCCEEDED", "FAILED"]
    files: tuple[PublicationJudgeArtifactFile, ...] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def file_set_is_exact(self) -> PublicationJudgeCriterionArtifacts:
        roles = tuple(item.role for item in self.files)
        if len(roles) != len(set(roles)):
            raise ValueError("Publication Judge criterion artifact roles must be unique")
        if len({item.path for item in self.files}) != len(self.files):
            raise ValueError("Publication Judge criterion artifact paths must be unique")
        prefix = f"criteria/{self.criterion_id.lower()}/"
        if any(not item.path.startswith(prefix) for item in self.files):
            raise ValueError("Publication Judge criterion artifact lies outside its directory")
        role_set = set(roles)
        if self.status == "SUCCEEDED" and role_set != {
            "INPUT",
            "RAW_OUTPUT",
            "RESULT",
            "METRICS",
        }:
            raise ValueError("Successful Judge criterion must seal all four artifacts")
        if self.status == "FAILED" and (
            not {"FAILURE", "METRICS"} <= role_set or "RESULT" in role_set
        ):
            raise ValueError("Failed Judge criterion must seal failure and metrics only")
        return self


class PublicationJudgeValidationReport(ManuscriptModel):
    schema_version: Literal[
        "manuscript-publication-judge-validation.v3",
        "manuscript-publication-judge-validation.v4",
        "manuscript-publication-judge-validation.v5",
        "manuscript-publication-judge-validation.v6",
        "manuscript-publication-judge-validation.v7",
        "manuscript-publication-judge-validation.v8",
    ]
    validation_id: str
    candidate_run_id: str
    package_id: Identifier
    profile_id: Identifier
    candidate_delivery_path: str = Field(min_length=1, max_length=1_000)
    candidate_delivery_sha256: Sha256
    fact_package_sha256: Sha256
    journal_profile_sha256: Sha256 | None = None
    writer_projection_sha256: Sha256
    claim_lineage_sha256: Sha256
    criterion_results: tuple[PublicationJudgeCriterionOutput, ...] = Field(max_length=7)
    failed_criteria: tuple[JudgeCriterionId, ...] = Field(max_length=7)
    criterion_artifacts: tuple[PublicationJudgeCriterionArtifacts, ...] = Field(
        default=(), max_length=7
    )
    weighted_score: float | None = Field(default=None, ge=0, le=100)
    critical_issue_count: int | None = Field(default=None, ge=0)
    limitations: tuple[
        Literal[
            "ROLE_ISOLATED_SAME_MODEL",
            "DIAGNOSTIC_NOT_HUMAN_ACCEPTANCE",
        ],
        Literal[
            "ROLE_ISOLATED_SAME_MODEL",
            "DIAGNOSTIC_NOT_HUMAN_ACCEPTANCE",
        ],
    ]

    @model_validator(mode="after")
    def result_partition_is_exact(self) -> PublicationJudgeValidationReport:
        result_ids = [item.criterion_id for item in self.criterion_results]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("Publication Judge criterion results must be unique")
        if set(result_ids) & set(self.failed_criteria):
            raise ValueError("Successful and failed Judge criteria overlap")
        if self.schema_version == "manuscript-publication-judge-validation.v3":
            if self.criterion_artifacts:
                raise ValueError("Historical v3 Judge report cannot carry v4 artifact seals")
        else:
            if self.journal_profile_sha256 is None:
                raise ValueError("Publication Judge v4 must bind the Journal Profile SHA-256")
            artifact_ids = [item.criterion_id for item in self.criterion_artifacts]
            if len(artifact_ids) != 7 or len(set(artifact_ids)) != 7:
                raise ValueError("Publication Judge v4 must seal exactly seven criteria")
            successful_artifacts = {
                item.criterion_id for item in self.criterion_artifacts if item.status == "SUCCEEDED"
            }
            failed_artifacts = {
                item.criterion_id for item in self.criterion_artifacts if item.status == "FAILED"
            }
            if successful_artifacts != set(result_ids) or failed_artifacts != set(
                self.failed_criteria
            ):
                raise ValueError("Publication Judge result partition differs from artifact seals")
        if self.schema_version in {
            "manuscript-publication-judge-validation.v3",
            "manuscript-publication-judge-validation.v4",
        }:
            if self.weighted_score is not None or self.critical_issue_count is not None:
                raise ValueError("Publication Judge historical validation schemas are qualitative")
            if any(
                item.rating is not None or item.critical_issue is not None
                for item in self.criterion_results
            ):
                raise ValueError(
                    "Publication Judge qualitative report cannot carry scored criteria"
                )
        elif self.schema_version == "manuscript-publication-judge-validation.v5":
            if any(
                item.schema_version != "manuscript-publication-judge-criterion.v4"
                for item in self.criterion_results
            ):
                raise ValueError("Publication Judge scored validation requires scored criteria")
            if any(
                item.rating_anchor is not None
                or item.manuscript_quote is not None
                or item.fact_quote is not None
                or item.improvement is not None
                for item in self.criterion_results
            ):
                raise ValueError("Publication Judge v5 cannot carry anchored score fields")
            if self.failed_criteria:
                if self.weighted_score is not None or self.critical_issue_count is not None:
                    raise ValueError(
                        "Publication Judge partial scored validation must keep score N/A"
                    )
            elif self.weighted_score is None or self.critical_issue_count is None:
                raise ValueError(
                    "Publication Judge complete scored validation must carry aggregates"
                )
        elif self.schema_version == "manuscript-publication-judge-validation.v6":
            if any(
                item.schema_version != "manuscript-publication-judge-criterion.v5"
                for item in self.criterion_results
            ):
                raise ValueError(
                    "Publication Judge anchored scored validation requires v5 criteria"
                )
            if self.failed_criteria:
                if self.weighted_score is not None or self.critical_issue_count is not None:
                    raise ValueError(
                        "Publication Judge partial anchored validation must keep score N/A"
                    )
            elif self.weighted_score is None or self.critical_issue_count is None:
                raise ValueError(
                    "Publication Judge complete anchored validation must carry aggregates"
                )
        elif self.schema_version == "manuscript-publication-judge-validation.v7":
            if any(
                item.schema_version != "manuscript-publication-judge-criterion.v6"
                for item in self.criterion_results
            ):
                raise ValueError(
                    "Publication Judge program-bound scored validation requires v6 criteria"
                )
            if self.failed_criteria:
                if self.weighted_score is not None or self.critical_issue_count is not None:
                    raise ValueError(
                        "Publication Judge partial program-bound validation must keep score N/A"
                    )
            elif self.weighted_score is None or self.critical_issue_count is None:
                raise ValueError(
                    "Publication Judge complete program-bound validation must carry aggregates"
                )
        else:
            if any(
                item.schema_version != "manuscript-publication-judge-criterion.v7"
                for item in self.criterion_results
            ):
                raise ValueError(
                    "Publication Judge normalized evidence validation requires v7 criteria"
                )
            if self.failed_criteria:
                if self.weighted_score is not None or self.critical_issue_count is not None:
                    raise ValueError(
                        "Publication Judge partial normalized validation must keep score N/A"
                    )
            elif self.weighted_score is None or self.critical_issue_count is None:
                raise ValueError(
                    "Publication Judge complete normalized validation must carry aggregates"
                )
        if self.limitations != (
            "ROLE_ISOLATED_SAME_MODEL",
            "DIAGNOSTIC_NOT_HUMAN_ACCEPTANCE",
        ):
            raise ValueError("Publication Judge limitations must remain explicit")
        return self


