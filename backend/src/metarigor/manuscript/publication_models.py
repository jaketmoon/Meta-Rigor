from __future__ import annotations

import hashlib
import re
from typing import Annotated, Literal
from pydantic import Field, model_validator
from pydantic_core import to_jsonable_python
from .models import Identifier, ManuscriptModel, ManuscriptReference, ManuscriptScope, MethodBasis, PackageIssue, Sha256, SourceAnchor, canonical_json
PublicationStatus = Literal[
    "REPORTED",
    "NOT_REPORTED",
    "NOT_APPLICABLE",
    "CONFLICT",
    "AWAITING_HUMAN_INPUT",
]


PublicationPlacement = Literal[
    "ABSTRACT", "MAIN_TEXT", "TABLE_FIGURE", "SUPPLEMENT", "EXPLICIT_DISPOSITION"
]


ProfileFit = Literal[
    "SUPPORTED_ARTICLE_TYPE",
    "CONDITIONAL_ARTICLE_TYPE",
    "PRESUBMISSION_INQUIRY_RECOMMENDED",
    "STYLE_ONLY",
]


def _english_surface(value: str) -> str:
    if re.search(r"[\u3400-\u9fff]", value):
        raise ValueError("Publication surface text must be English")
    return value


class PublicationArtifactBinding(ManuscriptModel):
    artifact_kind: Literal[
        "MANUSCRIPT_FACT_PACKAGE_V1",
        "ACCEPTED_PROTOCOL",
        "SOURCE_REVIEW",
        "REFERENCE_UPSTREAM",
        "EXTRACTION_VIEW",
        "HUMAN_INPUT",
    ]
    object_id: Identifier
    input_path: str = Field(min_length=1, max_length=10_000)
    expected_sha256: Sha256
    schema_version: str = Field(min_length=1, max_length=100)


class TerminalSourceBinding(ManuscriptModel):
    source_anchor_id: Identifier
    terminal_document_artifact_ref: Identifier
    terminal_document_sha256: Sha256
    terminal_view_artifact_ref: Identifier
    terminal_view_sha256: Sha256
    terminal_view_start_offset: int = Field(ge=0)
    terminal_view_end_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def terminal_view_span_is_ordered(self) -> TerminalSourceBinding:
        if self.terminal_view_end_offset <= self.terminal_view_start_offset:
            raise ValueError("Terminal Extraction View span must be ordered")
        return self


class PublicationFactCore(ManuscriptModel):
    fact_id: Identifier
    status: PublicationStatus
    statement_en: str = Field(min_length=1, max_length=8_000)
    required_for: tuple[Identifier, ...] = Field(min_length=1, max_length=100)
    allowed_placements: tuple[PublicationPlacement, ...] = Field(min_length=1, max_length=5)
    citation_keys: tuple[Identifier, ...] = Field(default=(), max_length=100)
    source_anchors: tuple[SourceAnchor, ...] = Field(default=(), max_length=100)
    terminal_source_bindings: tuple[TerminalSourceBinding, ...] = Field(default=(), max_length=100)
    upstream_artifact_refs: tuple[Identifier, ...] = Field(default=(), max_length=30)
    issue_refs: tuple[Identifier, ...] = Field(default=(), max_length=30)
    content_sha256: Sha256

    @model_validator(mode="after")
    def publication_fact_is_closed(self) -> PublicationFactCore:
        _english_surface(self.statement_en)
        if not self.source_anchors and not self.upstream_artifact_refs:
            raise ValueError("Publication Fact requires a source binding")
        if self.status in {"REPORTED", "CONFLICT"} and not self.source_anchors:
            raise ValueError("Reported or conflicting research facts require a Source Span")
        for values in (
            self.required_for,
            self.allowed_placements,
            self.citation_keys,
            self.upstream_artifact_refs,
            self.issue_refs,
        ):
            if len(values) != len(set(values)):
                raise ValueError("Publication Fact set-like fields must be unique")
        anchor_ids = {anchor.anchor_id for anchor in self.source_anchors}
        terminal_anchor_ids = [item.source_anchor_id for item in self.terminal_source_bindings]
        if len(terminal_anchor_ids) != len(set(terminal_anchor_ids)):
            raise ValueError("Terminal Source bindings must be unique by Source Anchor")
        if set(terminal_anchor_ids) - anchor_ids:
            raise ValueError("Terminal Source binding cites an unknown Source Anchor")
        if self.status in {"NOT_REPORTED", "CONFLICT", "AWAITING_HUMAN_INPUT"}:
            if not self.issue_refs:
                raise ValueError("Absent, conflicting, or human-owned facts require an Issue")
        accepted_hashes = {publication_fact_sha256(self)}
        if _only_backward_compatible_defaults_are_present(self):
            accepted_hashes.add(_legacy_publication_fact_sha256(self))
        if self.content_sha256 not in accepted_hashes:
            raise ValueError("Publication Fact content SHA-256 mismatch")
        return self


def publication_fact_sha256(value: PublicationFactCore | dict) -> str:
    if isinstance(value, ManuscriptModel):
        payload = value.model_dump(mode="json")
        # 新增的向后兼容字段只有在来源包显式携带时才进入 Fact hash；否则历史
        # immutable package 会因模型层注入默认值而发生伪变化。
        backward_compatible_defaults = {
            "p_value",
            "absolute_effect",
            "nnt",
            "is_primary",
            "assessment_level",
            "source_characteristics",
            "source_review_certainty",
            "source_review_certainty_rationale",
        }
        for field_name in backward_compatible_defaults - value.model_fields_set:
            payload.pop(field_name, None)
    else:
        payload = dict(value)
    payload.pop("content_sha256", None)
    payload = to_jsonable_python(payload)
    for anchor in payload.get("source_anchors", []):
        anchor.pop("input_path", None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _legacy_publication_fact_sha256(value: PublicationFactCore) -> str:
    payload = value.model_dump(mode="json")
    payload.pop("content_sha256", None)
    for field_name in {
        "p_value",
        "absolute_effect",
        "nnt",
        "is_primary",
        "assessment_level",
        "source_characteristics",
        "source_review_certainty",
        "source_review_certainty_rationale",
    }:
        payload.pop(field_name, None)
    for anchor in payload.get("source_anchors", []):
        anchor.pop("input_path", None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _only_backward_compatible_defaults_are_present(value: PublicationFactCore) -> bool:
    if getattr(value, "kind", None) == "SYNTHESIS":
        return (
            value.p_value is None
            and value.absolute_effect is None
            and value.nnt is None
            and value.is_primary is False
        )
    if getattr(value, "kind", None) == "RISK_OF_BIAS":
        return value.assessment_level == "STUDY_OUTCOME"
    if getattr(value, "kind", None) == "STUDY":
        return value.source_characteristics is None
    if getattr(value, "kind", None) == "CERTAINTY":
        return (
            value.source_review_certainty is None
            and value.source_review_certainty_rationale is None
        )
    return False


class NarrativePublicationFact(PublicationFactCore):
    kind: Literal["NARRATIVE"]
    topic: Literal[
        "REVIEW_IDENTITY",
        "OBJECTIVE",
        "ELIGIBILITY",
        "CONTEXT",
        "LIMITATION",
        "REGISTRATION",
        "SUPPORT",
        "AVAILABILITY",
        "METHOD_DEVIATION",
    ]


class OutcomePublicationFact(PublicationFactCore):
    kind: Literal["OUTCOME"]
    outcome_id: Identifier
    label: str = Field(min_length=1, max_length=1_000)
    definition: str = Field(min_length=1, max_length=4_000)
    timepoint: str = Field(min_length=1, max_length=1_000)
    role: Literal["PRIMARY", "SECONDARY", "SAFETY"]
    study_result_disposition: Literal["DE_TARGET", "REVIEW_SYNTHESIS_ONLY"]


class ReviewMethodPublicationFact(PublicationFactCore):
    kind: Literal["REVIEW_METHOD"]
    method_domain: Literal[
        "ELIGIBILITY",
        "SELECTION_PROCESS",
        "DATA_COLLECTION",
        "RISK_OF_BIAS",
        "EFFECT_MEASURE",
        "SYNTHESIS_MODEL",
        "HETEROGENEITY",
        "SUBGROUP",
        "SENSITIVITY",
        "REPORTING_BIAS",
        "CERTAINTY",
        "SOFTWARE",
    ]
    method_basis: MethodBasis
    description: str = Field(min_length=1, max_length=8_000)


class InterpretationPublicationFact(PublicationFactCore):
    kind: Literal["INTERPRETATION"]
    interpretation_kind: Literal[
        "PRINCIPAL_FINDING",
        "APPLICABILITY",
        "LIMITATION",
        "IMPLICATION",
        "COMPARISON_WITH_PRIOR_WORK",
    ]
    outcome_ids: tuple[Identifier, ...] = Field(default=(), max_length=100)


class SearchSourceFact(PublicationFactCore):
    kind: Literal["SEARCH_SOURCE"]
    source_name: str = Field(min_length=1, max_length=500)
    source_type: Literal["DATABASE", "REGISTER", "WEBSITE", "OTHER"]
    last_search_date: str | None = Field(default=None, max_length=100)
    strategy: str | None = Field(default=None, max_length=20_000)
    strategy_disposition: Literal["FULL", "SUPPLEMENT_BOUND", "NOT_REPORTED"]


class ExclusionReason(ManuscriptModel):
    reason: str = Field(min_length=1, max_length=500)
    report_count: int = Field(ge=1)


class SelectionFlowPublicationFact(PublicationFactCore):
    kind: Literal["SELECTION_FLOW"]
    records_identified: int | None = Field(default=None, ge=0)
    duplicates_removed: int | None = Field(default=None, ge=0)
    records_screened: int | None = Field(default=None, ge=0)
    reports_sought: int | None = Field(default=None, ge=0)
    reports_not_retrieved: int | None = Field(default=None, ge=0)
    reports_assessed: int | None = Field(default=None, ge=0)
    reports_excluded: int | None = Field(default=None, ge=0)
    studies_included: int = Field(ge=0)
    reports_included: int = Field(ge=0)
    exclusion_reasons: tuple[ExclusionReason, ...] = Field(default=(), max_length=100)


class StudyPublicationFact(PublicationFactCore):
    kind: Literal["STUDY"]
    study_id: Identifier
    report_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=50)
    design: str | None = Field(default=None, min_length=1, max_length=1_000)
    population: str | None = Field(default=None, min_length=1, max_length=2_000)
    intervention: str | None = Field(default=None, min_length=1, max_length=2_000)
    comparator: str | None = Field(default=None, min_length=1, max_length=2_000)
    sample_size: int | None = Field(default=None, ge=1)
    follow_up: str | None = Field(default=None, min_length=1, max_length=500)
    source_characteristics: str | None = Field(default=None, min_length=1, max_length=12_000)

    @model_validator(mode="after")
    def characteristics_match_status(self) -> StudyPublicationFact:
        values = (self.design, self.population, self.intervention, self.comparator, self.follow_up)
        if self.status == "REPORTED" and any(value is None for value in values):
            raise ValueError("A reported Study Fact requires source-bound characteristics")
        if self.status != "REPORTED" and any(value is not None for value in values):
            raise ValueError("Unavailable Study characteristics cannot contain inferred values")
        return self


class BinaryResultValues(ManuscriptModel):
    result_type: Literal["BINARY"]
    intervention_events: int = Field(ge=0)
    intervention_group_size: int = Field(ge=1)
    comparator_events: int = Field(ge=0)
    comparator_group_size: int = Field(ge=1)

    @model_validator(mode="after")
    def events_do_not_exceed_denominators(self) -> BinaryResultValues:
        if self.intervention_events > self.intervention_group_size:
            raise ValueError("Intervention events exceed denominator")
        if self.comparator_events > self.comparator_group_size:
            raise ValueError("Comparator events exceed denominator")
        return self


class ContinuousResultValues(ManuscriptModel):
    result_type: Literal["CONTINUOUS"]
    intervention_mean: float
    intervention_sd: float = Field(gt=0)
    intervention_group_size: int = Field(ge=2)
    comparator_mean: float
    comparator_sd: float = Field(gt=0)
    comparator_group_size: int = Field(ge=2)


StudyResultValues = Annotated[
    BinaryResultValues | ContinuousResultValues, Field(discriminator="result_type")
]


class StudyOutcomePublicationFact(PublicationFactCore):
    kind: Literal["STUDY_OUTCOME"]
    study_id: Identifier
    outcome_id: Identifier
    result_disposition: Literal[
        "EXTRACTED", "NOT_REPORTED", "SOURCE_UNAVAILABLE", "REFERENCE_CONFLICT"
    ]
    values: StudyResultValues | None
    reason: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def values_match_disposition(self) -> StudyOutcomePublicationFact:
        if (self.result_disposition == "EXTRACTED") != (self.values is not None):
            raise ValueError("Only EXTRACTED Study outcomes may contain values")
        if self.values is None and not self.reason:
            raise ValueError("Study outcome absence requires a reason")
        expected_status = {
            "EXTRACTED": "REPORTED",
            "NOT_REPORTED": "NOT_REPORTED",
            "SOURCE_UNAVAILABLE": "NOT_REPORTED",
            "REFERENCE_CONFLICT": "CONFLICT",
        }[self.result_disposition]
        if self.status != expected_status:
            raise ValueError("Study outcome status contradicts its result disposition")
        if self.result_disposition == "EXTRACTED":
            derived_anchor_ids = {
                anchor.anchor_id
                for anchor in self.source_anchors
                if anchor.source_representation == "CURATED_DERIVATION"
            }
            terminal_anchor_ids = {
                binding.source_anchor_id for binding in self.terminal_source_bindings
            }
            if derived_anchor_ids != terminal_anchor_ids:
                raise ValueError(
                    "Extracted derived evidence requires one terminal Source binding per anchor"
                )
        return self


class SynthesisPublicationFact(PublicationFactCore):
    kind: Literal["SYNTHESIS"]
    analysis_id: Identifier
    outcome_id: Identifier
    effect_measure: str = Field(min_length=1, max_length=100)
    estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    heterogeneity: str | None = Field(default=None, max_length=500)
    p_value: str | None = Field(default=None, max_length=100)
    absolute_effect: str | None = Field(default=None, max_length=1_000)
    nnt: str | None = Field(default=None, max_length=500)
    is_primary: bool = False
    contributing_study_ids: tuple[Identifier, ...] = Field(default=(), max_length=500)
    contributing_study_count: int | None = Field(default=None, ge=0)
    contributor_identity_disposition: Literal["FULL", "PARTIAL", "NOT_REPORTED"]
    analysis_population: str = Field(min_length=1, max_length=500)
    timepoint: str = Field(min_length=1, max_length=500)
    validity: Literal[
        "VALID",
        "REFERENCE_CONFLICT",
        "UNSUPPORTED_SYNTHESIS_METHOD",
        "INVALID_ESTIMAND",
        "NOT_REPORTED",
    ]
    # Phase 3 必须由 Run manifest 中独立创建并解析的 Human Decision 实现；
    # Fact Package 输入不能自行携带裁决并解锁冲突。
    human_adjudication: None = None

    @model_validator(mode="after")
    def synthesis_numbers_are_closed(self) -> SynthesisPublicationFact:
        numbers = (self.estimate, self.ci_lower, self.ci_upper)
        if any(value is None for value in numbers) and not all(value is None for value in numbers):
            raise ValueError("Synthesis estimate and confidence interval are an atomic set")
        if (
            self.ci_lower is not None
            and self.ci_upper is not None
            and self.ci_lower > self.ci_upper
        ):
            raise ValueError("Synthesis confidence interval is reversed")
        if self.validity == "VALID":
            if self.status != "REPORTED" or any(value is None for value in numbers):
                raise ValueError(
                    "A valid synthesis requires a reported estimate and confidence interval"
                )
        elif self.validity == "NOT_REPORTED":
            if (
                self.status != "NOT_REPORTED"
                or any(value is not None for value in numbers)
                or self.contributing_study_count not in {None, 0}
                or self.contributing_study_ids
                or self.contributor_identity_disposition != "NOT_REPORTED"
            ):
                raise ValueError("An unreported synthesis cannot contain inferred results")
        elif self.status != "CONFLICT":
            raise ValueError("A non-valid synthesis must remain a visible conflict")
        if len(self.contributing_study_ids) != len(set(self.contributing_study_ids)):
            raise ValueError("Contributing Study ids must be unique")
        if self.validity == "NOT_REPORTED":
            return self
        if self.contributor_identity_disposition == "FULL":
            if (
                self.contributing_study_count is None
                or len(self.contributing_study_ids) != self.contributing_study_count
            ):
                raise ValueError("Full contributor identity must match the reported Study count")
        elif self.contributor_identity_disposition == "PARTIAL":
            if (self.contributing_study_count is None and not self.contributing_study_ids) or (
                self.contributing_study_count is not None
                and len(self.contributing_study_ids) >= self.contributing_study_count
            ):
                raise ValueError("Partial contributor identity must remain incomplete")
        elif self.contributing_study_ids:
            raise ValueError("Unreported contributor identity cannot name Studies")
        return self


class AnalysisAvailabilityFact(PublicationFactCore):
    kind: Literal["ANALYSIS_AVAILABILITY"]
    analysis_kind: Literal["SENSITIVITY", "SUBGROUP", "REPORTING_BIAS", "META_REGRESSION"]
    outcome_id: Identifier | None
    disposition: Literal["REPORTED", "NOT_REPORTED", "NOT_APPLICABLE", "CONFLICT"]


class RiskOfBiasPublicationFact(PublicationFactCore):
    kind: Literal["RISK_OF_BIAS"]
    study_id: Identifier | None
    assessment_level: Literal["REVIEW_AGGREGATE", "STUDY", "STUDY_OUTCOME"] = "STUDY_OUTCOME"
    outcome_id: Identifier | None
    judgment: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def outcome_matches_assessment_level(self) -> RiskOfBiasPublicationFact:
        if self.assessment_level == "REVIEW_AGGREGATE":
            if self.study_id is not None or self.outcome_id is not None:
                raise ValueError("Aggregate risk of bias cannot name a Study or outcome")
            return self
        if self.study_id is None:
            raise ValueError("Study-level risk of bias requires a Study")
        if (self.assessment_level == "STUDY_OUTCOME") != (self.outcome_id is not None):
            raise ValueError("Risk-of-bias outcome must match its assessment level")
        return self


class CertaintyPublicationFact(PublicationFactCore):
    kind: Literal["CERTAINTY"]
    outcome_id: Identifier
    certainty: Literal["HIGH", "MODERATE", "LOW", "VERY_LOW", "NOT_ASSESSABLE"]
    rationale: str = Field(min_length=1, max_length=4_000)
    source_review_certainty: (
        Literal["HIGH", "MODERATE", "LOW", "VERY_LOW", "NOT_ASSESSABLE"] | None
    ) = None
    source_review_certainty_rationale: str | None = Field(
        default=None, min_length=1, max_length=4_000
    )

    @model_validator(mode="after")
    def review_certainty_fields_are_atomic(self) -> CertaintyPublicationFact:
        if (self.source_review_certainty is None) != (
            self.source_review_certainty_rationale is None
        ):
            raise ValueError("Source Review certainty and rationale are an atomic pair")
        return self


class AuthorResponsibilityFact(PublicationFactCore):
    kind: Literal["AUTHOR_RESPONSIBILITY"]
    field: Literal[
        "AUTHORSHIP",
        "AFFILIATIONS",
        "CONFLICTS_OF_INTEREST",
        "FUNDING",
        "AUTHOR_CONTRIBUTIONS",
        "DATA_SHARING",
        "AI_DISCLOSURE",
        "TRANSPARENCY_DECLARATION",
    ]
    value: str | None = Field(default=None, max_length=4_000)
    provided_by: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def responsibility_is_human_owned(self) -> AuthorResponsibilityFact:
        if self.status == "AWAITING_HUMAN_INPUT":
            if self.value is not None or self.provided_by is not None:
                raise ValueError("Awaiting author fields cannot contain invented values")
        elif self.value is None or self.provided_by is None:
            raise ValueError("Provided author fields require a named human source")
        return self


class PrismaChecklistDispositionFact(PublicationFactCore):
    kind: Literal["PRISMA_CHECKLIST_DISPOSITION"]
    guideline: Literal["PRISMA_2020", "PRISMA_S"]
    item_number: int = Field(ge=1, le=100)
    item_label: str = Field(min_length=1, max_length=500)


PublicationFact = Annotated[
    NarrativePublicationFact
    | OutcomePublicationFact
    | ReviewMethodPublicationFact
    | InterpretationPublicationFact
    | SearchSourceFact
    | SelectionFlowPublicationFact
    | StudyPublicationFact
    | StudyOutcomePublicationFact
    | SynthesisPublicationFact
    | AnalysisAvailabilityFact
    | RiskOfBiasPublicationFact
    | CertaintyPublicationFact
    | AuthorResponsibilityFact
    | PrismaChecklistDispositionFact,
    Field(discriminator="kind"),
]


class PublicationCurationRecord(ManuscriptModel):
    curator: str = Field(min_length=1, max_length=200)
    curated_at: str = Field(min_length=1, max_length=100)
    tool_version: Literal["metarigor-manuscript-curator-v2"]
    input_sha256s: tuple[Sha256, ...] = Field(min_length=1, max_length=500)
    notes: str = Field(min_length=1, max_length=4_000)
    review_status: Literal["CURATED_AND_HASH_VERIFIED"]


class ManuscriptFactPackageV2(ManuscriptModel):
    schema_version: Literal["manuscript-fact-package.v2"]
    package_id: Identifier
    case_id: Identifier
    language: Literal["en"]
    source_package_v1: PublicationArtifactBinding
    research_artifacts: tuple[PublicationArtifactBinding, ...] = Field(min_length=1, max_length=100)
    scope: ManuscriptScope
    study_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1_000)
    outcome_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=1_000)
    study_result_outcome_ids: tuple[Identifier, ...] | None = Field(
        default=None, min_length=1, max_length=1_000
    )
    fact_scope: Literal[
        "HEADLINE_ONLY", "COMPLETE_SOURCE_SUPPORTED", "REFERENCE_UPSTREAM_PROXY"
    ] = "HEADLINE_ONLY"
    required_slots: tuple[Identifier, ...] = Field(min_length=1, max_length=5_000)
    facts: tuple[PublicationFact, ...] = Field(min_length=1, max_length=20_000)
    references: tuple[ManuscriptReference, ...] = Field(default=(), max_length=2_000)
    issues: tuple[PackageIssue, ...] = Field(default=(), max_length=5_000)
    curation: PublicationCurationRecord

    @model_validator(mode="after")
    def publication_package_is_closed(self) -> ManuscriptFactPackageV2:
        if len(self.study_ids) != len(set(self.study_ids)):
            raise ValueError("Study ids must be unique")
        if len(self.outcome_ids) != len(set(self.outcome_ids)):
            raise ValueError("Outcome ids must be unique")
        cell_outcome_ids = self.study_result_outcome_ids or self.outcome_ids
        if len(cell_outcome_ids) != len(set(cell_outcome_ids)):
            raise ValueError("Study-result outcome ids must be unique")
        if set(cell_outcome_ids) - set(self.outcome_ids):
            raise ValueError("Study-result outcome lies outside the outcome inventory")
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("Publication Fact ids must be unique")
        issue_ids = {issue.issue_id for issue in self.issues}
        reference_keys = {reference.citation_key for reference in self.references}
        artifact_ids = {
            self.source_package_v1.object_id,
            *(binding.object_id for binding in self.research_artifacts),
        }
        artifact_by_id = {
            self.source_package_v1.object_id: self.source_package_v1,
            **{binding.object_id: binding for binding in self.research_artifacts},
        }
        artifact_hashes = {binding.expected_sha256 for binding in artifact_by_id.values()}
        for fact in self.facts:
            if set(fact.issue_refs) - issue_ids:
                raise ValueError("Publication Fact cites an unknown Issue")
            if set(fact.citation_keys) - reference_keys:
                raise ValueError("Publication Fact cites an unknown reference")
            if set(fact.upstream_artifact_refs) - artifact_ids:
                raise ValueError("Publication Fact cites an unknown upstream artifact")
            if any(anchor.document_sha256 not in artifact_hashes for anchor in fact.source_anchors):
                raise ValueError("Publication Fact Source Anchor lacks a bound Source Document")
            for terminal in fact.terminal_source_bindings:
                document = artifact_by_id.get(terminal.terminal_document_artifact_ref)
                if (
                    document is None
                    or document.artifact_kind != "REFERENCE_UPSTREAM"
                    or document.expected_sha256 != terminal.terminal_document_sha256
                ):
                    raise ValueError(
                        "Derived Source Anchor lacks its exact terminal Source Document"
                    )
                view = artifact_by_id.get(terminal.terminal_view_artifact_ref)
                if (
                    view is None
                    or view.artifact_kind != "EXTRACTION_VIEW"
                    or view.expected_sha256 != terminal.terminal_view_sha256
                ):
                    raise ValueError(
                        "Derived Source Anchor lacks its exact terminal Extraction View"
                    )
            if fact.kind == "STUDY_OUTCOME" and fact.status == "REPORTED":
                if {item.source_anchor_id for item in fact.terminal_source_bindings} != {
                    item.anchor_id for item in fact.source_anchors
                }:
                    raise ValueError(
                        "Every reported Study Outcome Source Anchor requires a terminal binding"
                    )
        for slot in self.required_slots:
            owners = [fact for fact in self.facts if slot in fact.required_for]
            if len(owners) != 1:
                raise ValueError(f"Required publication slot must have one owner: {slot}")
        study_fact_ids = [fact.study_id for fact in self.facts if fact.kind == "STUDY"]
        if len(study_fact_ids) != len(set(study_fact_ids)):
            raise ValueError("Each Study must have exactly one Study Fact")
        if set(study_fact_ids) != set(self.study_ids):
            raise ValueError("Study inventory and Study Facts differ")
        outcome_cell_list = [
            (fact.study_id, fact.outcome_id) for fact in self.facts if fact.kind == "STUDY_OUTCOME"
        ]
        if len(outcome_cell_list) != len(set(outcome_cell_list)):
            raise ValueError("Each Study x Outcome cell must have exactly one fact")
        outcome_cells = set(outcome_cell_list)
        expected_cells = {
            (study, outcome) for study in self.study_ids for outcome in cell_outcome_ids
        }
        if outcome_cells != expected_cells:
            raise ValueError("Study x Outcome publication closure is incomplete")
        aggregate_rob = [
            fact
            for fact in self.facts
            if fact.kind == "RISK_OF_BIAS" and fact.assessment_level == "REVIEW_AGGREGATE"
        ]
        study_outcome_rob_cells = [
            (fact.study_id, fact.outcome_id)
            for fact in self.facts
            if fact.kind == "RISK_OF_BIAS" and fact.assessment_level == "STUDY_OUTCOME"
        ]
        study_rob_cells = [
            fact.study_id
            for fact in self.facts
            if fact.kind == "RISK_OF_BIAS" and fact.assessment_level == "STUDY"
        ]
        if aggregate_rob:
            if len(aggregate_rob) != 1 or study_rob_cells or study_outcome_rob_cells:
                raise ValueError(
                    "Review-aggregate risk-of-bias closure must contain exactly one fact"
                )
        elif study_rob_cells:
            if study_outcome_rob_cells:
                raise ValueError("Risk-of-bias closure cannot mix Study and Study-outcome levels")
            if len(study_rob_cells) != len(set(study_rob_cells)) or set(study_rob_cells) != set(
                self.study_ids
            ):
                raise ValueError("Study-level risk-of-bias closure must match the Study inventory")
        elif (
            len(study_outcome_rob_cells) != len(set(study_outcome_rob_cells))
            or set(study_outcome_rob_cells) != expected_cells
        ):
            raise ValueError("Result-specific risk-of-bias closure must match Study x Outcome")
        certainty_outcomes = [fact.outcome_id for fact in self.facts if fact.kind == "CERTAINTY"]
        if len(certainty_outcomes) != len(set(certainty_outcomes)):
            raise ValueError("Each outcome must have exactly one certainty fact")
        if set(certainty_outcomes) != set(self.outcome_ids):
            raise ValueError("Outcome-specific certainty closure is incomplete")
        for fact in self.facts:
            if fact.kind in {"STUDY_OUTCOME", "RISK_OF_BIAS"}:
                if fact.kind == "RISK_OF_BIAS" and fact.assessment_level == "REVIEW_AGGREGATE":
                    continue
                if fact.study_id not in self.study_ids or (
                    fact.outcome_id is not None and fact.outcome_id not in self.outcome_ids
                ):
                    raise ValueError("Fact references a Study or outcome outside the inventory")
            elif fact.kind == "CERTAINTY" and fact.outcome_id not in self.outcome_ids:
                raise ValueError("Certainty references an outcome outside the inventory")
            elif fact.kind == "SYNTHESIS":
                if fact.outcome_id not in self.outcome_ids:
                    raise ValueError("Synthesis references an outcome outside the inventory")
                if set(fact.contributing_study_ids) - set(self.study_ids):
                    raise ValueError("Synthesis contributor lies outside the Study inventory")
            elif fact.kind == "OUTCOME" and fact.outcome_id not in self.outcome_ids:
                raise ValueError("Outcome Fact references an outcome outside the inventory")
            elif fact.kind == "INTERPRETATION" and set(fact.outcome_ids) - set(self.outcome_ids):
                raise ValueError("Interpretation references an outcome outside the inventory")
        if self.fact_scope in {"COMPLETE_SOURCE_SUPPORTED", "REFERENCE_UPSTREAM_PROXY"}:
            outcome_fact_ids = [fact.outcome_id for fact in self.facts if fact.kind == "OUTCOME"]
            if len(outcome_fact_ids) != len(set(outcome_fact_ids)) or set(outcome_fact_ids) != set(
                self.outcome_ids
            ):
                raise ValueError("Complete Fact Package requires one definition per outcome")
            synthesis_ids = [fact.analysis_id for fact in self.facts if fact.kind == "SYNTHESIS"]
            if len(synthesis_ids) != len(set(synthesis_ids)):
                raise ValueError("Synthesis analysis ids must be unique")
            synthesis_outcomes = [
                fact.outcome_id for fact in self.facts if fact.kind == "SYNTHESIS"
            ]
            if len(synthesis_outcomes) != len(set(synthesis_outcomes)) or set(
                synthesis_outcomes
            ) != set(self.outcome_ids):
                raise ValueError("Complete Fact Package requires one synthesis per outcome")
            if sum(fact.is_primary for fact in self.facts if fact.kind == "SYNTHESIS") != 1:
                raise ValueError("Complete Fact Package requires exactly one primary synthesis")
            method_domain_list = [
                fact.method_domain for fact in self.facts if fact.kind == "REVIEW_METHOD"
            ]
            method_domains = set(method_domain_list)
            required_method_domains = {
                "ELIGIBILITY",
                "SELECTION_PROCESS",
                "DATA_COLLECTION",
                "RISK_OF_BIAS",
                "EFFECT_MEASURE",
                "SYNTHESIS_MODEL",
                "HETEROGENEITY",
                "SUBGROUP",
                "SENSITIVITY",
                "REPORTING_BIAS",
                "CERTAINTY",
                "SOFTWARE",
            }
            if method_domains != required_method_domains or len(method_domain_list) != len(
                required_method_domains
            ):
                raise ValueError(
                    "Complete Fact Package requires exactly one Fact per Review method domain"
                )
        return self


class ProfileSourceBinding(ManuscriptModel):
    source_id: Identifier
    official_url: str = Field(pattern=r"^https://", max_length=2_000)
    retrieved_at: str = Field(min_length=10, max_length=100)
    applicable_from: str | None = Field(default=None, max_length=100)
    snapshot_path: str = Field(min_length=1, max_length=1_000)
    snapshot_sha256: Sha256
    supports: tuple[Identifier, ...] = Field(min_length=1, max_length=30)


class WordBudget(ManuscriptModel):
    maximum: int | None = Field(default=None, ge=100)
    excludes: tuple[
        Literal["ABSTRACT"],
        Literal["REFERENCES"],
        Literal["TABLES"],
        Literal["FIGURES"],
        Literal["SUPPLEMENT"],
    ]


class AbstractContract(ManuscriptModel):
    structured: Literal[True]
    headings: tuple[Identifier, ...] = Field(min_length=1, max_length=20)
    minimum_words: int | None = Field(default=None, ge=50)
    maximum_words: int | None = Field(default=None, ge=50)

    @model_validator(mode="after")
    def abstract_range_is_ordered(self) -> AbstractContract:
        if (
            self.minimum_words is not None
            and self.maximum_words is not None
            and self.minimum_words > self.maximum_words
        ):
            raise ValueError("Abstract word range is reversed")
        return self


class SectionContract(ManuscriptModel):
    section_id: Literal["INTRODUCTION", "METHODS", "RESULTS", "DISCUSSION"]
    heading: str = Field(min_length=1, max_length=200)
    order: int = Field(ge=1)


class StatementContract(ManuscriptModel):
    statement_id: Identifier
    author_field: str | None = Field(default=None, max_length=100)
    placement: PublicationPlacement


class FigureTableBudget(ManuscriptModel):
    maximum_total: int | None = Field(default=None, ge=1)
    required_objects: tuple[Identifier, ...] = Field(default=(), max_length=30)


class SupplementContract(ManuscriptModel):
    required_modules: tuple[Identifier, ...] = Field(default=(), max_length=50)
    prisma_flow_placement: Literal["MAIN", "SUPPLEMENT"]


class TitleContract(ManuscriptModel):
    suffix: str = Field(min_length=1, max_length=200)


class JournalPresentationProfile(ManuscriptModel):
    schema_version: Literal["manuscript-journal-profile.v1"]
    profile_id: Identifier
    profile_version: str = Field(min_length=1, max_length=100)
    journal_name: str = Field(min_length=1, max_length=200)
    article_type: str = Field(min_length=1, max_length=200)
    submission_fit: ProfileFit
    source_bindings: tuple[ProfileSourceBinding, ...] = Field(min_length=1, max_length=20)
    reporting_guidelines: tuple[Identifier, ...] = Field(min_length=1, max_length=20)
    title_contract: TitleContract
    word_budget: WordBudget
    abstract_contract: AbstractContract
    front_matter_modules: tuple[Identifier, ...] = Field(default=(), max_length=20)
    section_contract: tuple[SectionContract, ...] = Field(min_length=4, max_length=30)
    required_statements: tuple[StatementContract, ...] = Field(default=(), max_length=30)
    main_figure_table_budget: FigureTableBudget
    supplement_contract: SupplementContract
    citation_style: Literal["SOURCE_BOUND_KEYED"]
    reference_minimum: int | None = Field(default=None, ge=1)
    reference_maximum: int | None = Field(default=None, ge=1)
    forbidden_claims: tuple[str, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def profile_is_presentation_only(self) -> JournalPresentationProfile:
        sections = sorted(self.section_contract, key=lambda value: value.order)
        if [section.order for section in sections] != list(range(1, len(sections) + 1)):
            raise ValueError("Profile section order must be contiguous")
        if len({section.section_id for section in sections}) != len(sections):
            raise ValueError("Profile section ids must be unique")
        if tuple(section.section_id for section in sections) != (
            "INTRODUCTION",
            "METHODS",
            "RESULTS",
            "DISCUSSION",
        ):
            raise ValueError("Wave 1 Profile sections must use the supported IMRaD contract")
        expected_exclusions = (
            "ABSTRACT",
            "REFERENCES",
            "TABLES",
            "FIGURES",
            "SUPPLEMENT",
        )
        if self.word_budget.excludes != expected_exclusions:
            raise ValueError("Wave 1 main-text word-budget exclusions are fixed")
        supported_front = {"KEY_POINTS", "WHAT_IS_ALREADY_KNOWN", "WHAT_THIS_STUDY_ADDS"}
        if set(self.front_matter_modules) - supported_front:
            raise ValueError("Profile requests an unsupported front matter module")
        supported_supplement = {
            "PRISMA_FLOW",
            "SEARCH_STRATEGIES",
            "STUDY_OUTCOME_INVENTORY",
            "ADDITIONAL_ANALYSIS_DISPOSITIONS",
            "AUTHOR_INPUT_DISPOSITIONS",
        }
        if set(self.supplement_contract.required_modules) - supported_supplement:
            raise ValueError("Profile requests an unsupported supplement module")
        supported_objects = {
            "PRISMA_FLOW",
            "STUDY_CHARACTERISTICS",
            "RISK_OF_BIAS",
            "FOREST_PLOT",
            "SUMMARY_OF_FINDINGS",
        }
        if set(self.main_figure_table_budget.required_objects) - supported_objects:
            raise ValueError("Profile requests an unsupported table or figure")
        supported_abstract = {
            "BACKGROUND",
            "IMPORTANCE",
            "OBJECTIVE",
            "METHODS",
            "DATA_SOURCES",
            "STUDY_SELECTION",
            "DATA_EXTRACTION_AND_SYNTHESIS",
            "MAIN_OUTCOMES_AND_MEASURES",
            "RESULTS",
            "LIMITATIONS",
            "CONCLUSIONS",
            "CONCLUSIONS_AND_RELEVANCE",
            "DESIGN",
            "ELIGIBILITY_CRITERIA",
            "MAIN_OUTCOME_MEASURES",
            "REGISTRATION",
        }
        if set(self.abstract_contract.headings) - supported_abstract:
            raise ValueError("Profile requests an unsupported abstract heading")
        supported_requirements = {
            requirement for binding in self.source_bindings for requirement in binding.supports
        }
        required_source_contracts = {
            "ARTICLE_TYPE",
            *self.reporting_guidelines,
        }
        if required_source_contracts - supported_requirements:
            raise ValueError("Profile requirements lack an official source binding")
        serialized = canonical_json(self).decode("utf-8").lower()
        forbidden_research_keys = ('"estimate"', '"ci_lower"', '"study_ids"', '"outcome_ids"')
        if any(key in serialized for key in forbidden_research_keys):
            raise ValueError("Journal Profile cannot contain research facts")
        return self


class BoundInput(ManuscriptModel):
    input_path: str = Field(min_length=1, max_length=10_000)
    expected_sha256: Sha256


class ManuscriptRunRequestV2(ManuscriptModel):
    schema_version: Literal["manuscript-run-request.v2"]
    run_id: str = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )
    fact_package: BoundInput
    profile_id: Identifier
    journal_profile: BoundInput
    writer_mode: Literal["DETERMINISTIC", "LLM"] = "DETERMINISTIC"
    reference_package: BoundInput | None = None
    input_curation: BoundInput | None = None

    @model_validator(mode="after")
    def llm_profile_is_explicit(self) -> ManuscriptRunRequestV2:
        if self.writer_mode == "LLM" and self.profile_id != "generic_prisma_2020_meta_analysis_v2":
            raise ValueError("LLM writer currently requires the Generic V2 Profile")
        if self.writer_mode == "LLM" and (
            self.reference_package is None or self.input_curation is None
        ):
            raise ValueError("LLM writer requires its reference package and curation audit")
        return self


class CoverageItem(ManuscriptModel):
    requirement_id: Identifier
    requirement_source: Literal["FACT_PACKAGE", "PRISMA_2020", "PRISMA_S", "JOURNAL_PROFILE"]
    fact_ids: tuple[Identifier, ...]
    placement: PublicationPlacement
    artifact_path: str = Field(min_length=1, max_length=1_000)
    section_id: Identifier | None
    disposition: Literal["COVERED", "EXPLICIT_ABSENCE", "CONFLICT_VISIBLE"]
    status: Literal["PASSED", "FAILED", "UNVERIFIED"]


class CoverageManifest(ManuscriptModel):
    schema_version: Literal["manuscript-coverage.v1"]
    package_id: Identifier
    profile_id: Identifier
    items: tuple[CoverageItem, ...]
    required_count: int = Field(ge=0)
    covered_count: int = Field(ge=0)


class ComplianceCheck(ManuscriptModel):
    check_id: Identifier
    status: Literal["PASSED", "FAILED", "NOT_APPLICABLE", "AWAITING_HUMAN_INPUT"]
    detail: str = Field(min_length=1, max_length=4_000)


class ProfileComplianceReport(ManuscriptModel):
    schema_version: Literal["manuscript-profile-compliance.v1"]
    package_id: Identifier
    profile_id: Identifier
    profile_sha256: Sha256
    checks: tuple[ComplianceCheck, ...]
    status: Literal["PROFILE_COMPLETE", "NOT_READY_FOR_HUMAN_FINALIZATION", "PROFILE_NONCOMPLIANT"]


class PublicationVerificationIssue(ManuscriptModel):
    category: str = Field(min_length=1, max_length=200)
    severity: Literal["WARNING", "ERROR"]
    detail: str = Field(min_length=1, max_length=4_000)
    fact_ids: tuple[Identifier, ...] = Field(default=(), max_length=100)


class PublicationVerificationReport(ManuscriptModel):
    schema_version: Literal["manuscript-publication-verifier.v1"]
    package_id: Identifier
    profile_id: Identifier
    verdict: Literal["PASSED", "ISSUES_FOUND"]
    issues: tuple[PublicationVerificationIssue, ...]
    checked_fact_count: int = Field(ge=0)
    coverage_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)


class PublicationDeliveryFile(ManuscriptModel):
    role: str = Field(min_length=1, max_length=100)
    path: str = Field(min_length=1, max_length=1_000)
    sha256: Sha256


class ManuscriptDeliveryV2(ManuscriptModel):
    schema_version: Literal["manuscript-delivery.v2"]
    run_id: str
    package_id: Identifier
    profile_id: Identifier
    profile_sha256: Sha256
    files: tuple[PublicationDeliveryFile, ...] = Field(min_length=1, max_length=100)
    issue_count: int = Field(ge=0)
    readiness: Literal[
        "READY_FOR_HUMAN_FINALIZATION",
        "NOT_READY_FOR_HUMAN_FINALIZATION",
        "AUDIT_ONLY",
    ]


class ManuscriptRunResultV2(ManuscriptModel):
    schema_version: Literal["manuscript-run-result.v2"]
    run_id: str
    package_id: Identifier
    profile_id: Identifier
    status: Literal["SUCCEEDED", "COMPLETED_WITH_ISSUES", "FAILED"]
    issue_count: int = Field(ge=0)
    delivery_path: str | None
    delivery_sha256: Sha256 | None


