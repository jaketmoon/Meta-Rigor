from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


GradeDomain = Literal[
    "RISK_OF_BIAS",
    "INCONSISTENCY",
    "INDIRECTNESS",
    "IMPRECISION",
    "PUBLICATION_BIAS",
]


DomainJudgment = Literal[
    "NOT_SERIOUS",
    "BORDERLINE_NOT_SERIOUS_SERIOUS",
    "SERIOUS",
    "BORDERLINE_SERIOUS_VERY_SERIOUS",
    "VERY_SERIOUS",
]


Certainty = Literal["HIGH", "MODERATE", "LOW", "VERY_LOW"]


StudyRiskOfBiasJudgment = Literal["LOW", "SOME_CONCERNS", "HIGH", "NOT_ASSESSED"]


EvidenceFactKind = Literal[
    "PROTOCOL_TARGET",
    "SYNTHESIS_RESULT",
    "STUDY_DESIGN",
    "RISK_OF_BIAS",
    "INCONSISTENCY",
    "INDIRECTNESS",
    "IMPRECISION",
    "PUBLICATION_BIAS",
    "SOURCE_LIMITATION",
]


SourceRepresentation = Literal["RAW_SOURCE", "CURATED_DERIVATION"]


ComparisonKind = Literal["DIRECT_TWO_ARM", "MULTI_ARM", "NETWORK", "OTHER"]


SynthesisKind = Literal[
    "PAIRWISE_META_ANALYSIS",
    "SINGLE_STUDY",
    "SUBGROUP_ANALYSIS",
    "SENSITIVITY_ANALYSIS",
    "OTHER",
]


OutcomeKind = Literal["BINARY", "CONTINUOUS", "TIME_TO_EVENT", "ORDINAL", "OTHER"]


ScopeBasisField = Literal[
    "selected_outcome_ids",
    "review_role",
    "comparison_kind",
    "synthesis_kind",
    "outcome_kind",
]


StudyDesignSupportField = Literal[
    "PARTICIPANT_RANDOMIZATION",
    "EXACTLY_TWO_FIXED_ARMS",
    "PARALLEL_ASSIGNMENT_CONTEXT",
]


class EvidenceCertaintyModel(BaseModel):
    """Evidence Certainty 的所有公开、私有和模型契约都拒绝额外字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self)).hexdigest()


def canonical_json(value: BaseModel | dict) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class SourceDocumentReference(EvidenceCertaintyModel):
    """派生 Source Span 指向的上游原始文档；不是新的研究事实。"""

    document_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    input_path: str = Field(min_length=1, max_length=10_000)
    document_sha256: Sha256
    source_representation: Literal["RAW_SOURCE"] = "RAW_SOURCE"
    locator_kind: Literal["TEXT_SECTION", "FIGURE_REGION", "TABLE_ROWS", "PDF_PAGES"]
    locator: str = Field(min_length=1, max_length=2_000)
    derivation: Literal[
        "DIRECT_TRANSCRIPTION",
        "FIGURE_TRANSCRIPTION",
        "CURATOR_RECONCILIATION",
        "SCOPE_CLOSURE_AUDIT",
        "DETERMINISTIC_PDF_TEXT_EXTRACTION",
    ]
    output_fields: tuple[str, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def output_fields_are_unique(self) -> SourceDocumentReference:
        if len(self.output_fields) != len(set(self.output_fields)):
            raise ValueError("Source Document reference output_fields must be unique")
        return self


class SourceSpan(EvidenceCertaintyModel):
    span_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,159}$")
    document_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    input_path: str = Field(min_length=1, max_length=10_000)
    document_sha256: Sha256
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=20_000)
    quote_sha256: Sha256
    source_representation: SourceRepresentation
    derivation_sources: tuple[SourceDocumentReference, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def quote_binding_is_closed(self) -> SourceSpan:
        if self.end_offset <= self.start_offset:
            raise ValueError("Source Span end_offset must be greater than start_offset")
        if self.end_offset - self.start_offset != len(self.quote):
            raise ValueError("Source Span offsets must match quote character length")
        observed = hashlib.sha256(self.quote.encode("utf-8")).hexdigest()
        if observed != self.quote_sha256:
            raise ValueError("Source Span quote_sha256 does not match quote")
        identities = [
            (source.document_id, source.document_sha256, source.locator)
            for source in self.derivation_sources
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Source Span derivation sources must be unique")
        if any(source.document_id == self.document_id for source in self.derivation_sources):
            raise ValueError("Source Span cannot cite itself as an upstream derivation source")
        if self.source_representation == "RAW_SOURCE" and self.derivation_sources:
            raise ValueError("raw Source Span cannot carry derivation lineage")
        if self.source_representation == "CURATED_DERIVATION" and not self.derivation_sources:
            raise ValueError("curated Source Span requires one-hop raw source lineage")
        return self


class EvidenceFact(EvidenceCertaintyModel):
    fact_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,159}$")
    fact_kind: EvidenceFactKind
    statement: str = Field(min_length=1, max_length=8_000)
    domain_relevance: tuple[GradeDomain, ...] = Field(min_length=1, max_length=5)
    source_span: SourceSpan

    @model_validator(mode="after")
    def domains_are_unique(self) -> EvidenceFact:
        if len(self.domain_relevance) != len(set(self.domain_relevance)):
            raise ValueError("Evidence Fact domain_relevance must be unique")
        return self


class ContributingStudy(EvidenceCertaintyModel):
    study_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=1_000)
    source_fact_ids: tuple[str, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def source_fact_ids_are_unique(self) -> ContributingStudy:
        if len(self.source_fact_ids) != len(set(self.source_fact_ids)):
            raise ValueError("Contributing Study source_fact_ids must be unique")
        return self


class ProtocolTargetContext(EvidenceCertaintyModel):
    review_question: str = Field(min_length=1, max_length=4_000)
    eligibility_criteria: tuple[str, ...] = Field(min_length=1, max_length=100)


class StudyRiskOfBiasProfile(EvidenceCertaintyModel):
    source_fact_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,159}$")
    study_judgments: dict[str, StudyRiskOfBiasJudgment] = Field(min_length=1)
    contribution_weight_percent: dict[str, float] = Field(default_factory=dict)
    observations: tuple[str, ...] = Field(min_length=1, max_length=100)

    @property
    def source_binding_quote(self) -> str:
        payload = self.model_dump(mode="json", exclude={"source_fact_id"})
        return f"risk_of_bias_profile={canonical_json(payload).decode('utf-8')}"

    @model_validator(mode="after")
    def weights_are_complete_when_present(self) -> StudyRiskOfBiasProfile:
        if self.contribution_weight_percent and set(self.contribution_weight_percent) != set(
            self.study_judgments
        ):
            raise ValueError("risk-of-bias contribution weights must be complete or absent")
        if any(not 0 <= value <= 100 for value in self.contribution_weight_percent.values()):
            raise ValueError("risk-of-bias contribution weights must be percentages")
        if (
            self.contribution_weight_percent
            and abs(sum(self.contribution_weight_percent.values()) - 100) > 0.05
        ):
            raise ValueError("risk-of-bias contribution weights must sum to 100 percent")
        return self


class EvidencePicoProfile(EvidenceCertaintyModel):
    source_fact_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,159}$")
    population: str = Field(min_length=1, max_length=4_000)
    intervention: str = Field(min_length=1, max_length=2_000)
    comparator: str = Field(min_length=1, max_length=2_000)
    outcome: str = Field(min_length=1, max_length=2_000)
    timepoint: str = Field(min_length=1, max_length=1_000)
    setting: str = Field(min_length=1, max_length=2_000)
    estimand: str = Field(min_length=1, max_length=2_000)
    notable_variations: tuple[str, ...] = Field(default=(), max_length=100)

    @property
    def source_binding_quote(self) -> str:
        payload = self.model_dump(mode="json", exclude={"source_fact_id"})
        return f"evidence_pico={canonical_json(payload).decode('utf-8')}"


class PublishedSynthesisResult(EvidenceCertaintyModel):
    source_fact_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,159}$")
    effect_measure: Literal["RR", "OR", "RD", "MD", "SMD"]
    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: Literal[95] = 95
    study_count: int = Field(ge=1, le=10_000)
    participant_count: int = Field(ge=1)
    event_count: int | None = Field(default=None, ge=0)
    control_event_count: int | None = Field(default=None, ge=0)
    control_participant_count: int | None = Field(default=None, ge=1)
    i2_percent: float | None = Field(default=None, ge=0, le=100)
    prediction_interval_lower: float | None = None
    prediction_interval_upper: float | None = None
    analysis_method: str = Field(min_length=1, max_length=2_000)

    @property
    def source_binding_quote(self) -> str:
        """完整字段的确定性 transcription；必须由绑定 Source Span 逐字承载。"""

        payload = self.model_dump(mode="json", exclude={"source_fact_id"})
        return f"synthesis={canonical_json(payload).decode('utf-8')}"

    @model_validator(mode="after")
    def interval_is_valid(self) -> PublishedSynthesisResult:
        if not self.ci_lower <= self.estimate <= self.ci_upper:
            raise ValueError("Published synthesis estimate must lie inside its CI")
        if self.effect_measure in {"RR", "OR"} and self.ci_lower <= 0:
            raise ValueError("ratio effect and CI must be positive")
        pair = (self.prediction_interval_lower, self.prediction_interval_upper)
        if (pair[0] is None) != (pair[1] is None):
            raise ValueError("prediction interval requires both bounds")
        if pair[0] is not None and pair[0] > pair[1]:
            raise ValueError("prediction interval bounds are reversed")
        control = (self.control_event_count, self.control_participant_count)
        if (control[0] is None) != (control[1] is None):
            raise ValueError("binary control risk requires events and participants")
        if control[0] is not None and control[0] > control[1]:
            raise ValueError("control events cannot exceed control participants")
        if self.event_count is not None and self.event_count > self.participant_count:
            raise ValueError("events cannot exceed total participants")
        if control[1] is not None and control[1] > self.participant_count:
            raise ValueError("control participants cannot exceed total participants")
        if control[0] is not None and self.effect_measure not in {"RR", "OR", "RD"}:
            raise ValueError("control event risk is only valid for binary effects")
        return self


class CertaintyTarget(EvidenceCertaintyModel):
    target_kind: Literal["NULL_EFFECT"] = "NULL_EFFECT"
    threshold_value: Literal[0.0, 1.0]
    interpretation: str = Field(min_length=1, max_length=2_000)
    threshold_source_fact_id: None = None

    @model_validator(mode="after")
    def null_target_is_frozen(self) -> CertaintyTarget:
        if self.target_kind != "NULL_EFFECT":
            raise ValueError("EC V1 freezes the certainty target to NULL_EFFECT")
        return self


class ScopeBasis(EvidenceCertaintyModel):
    selected_outcome_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    review_role: Literal["HEADLINE"] = "HEADLINE"
    comparison_kind: ComparisonKind
    synthesis_kind: SynthesisKind
    outcome_kind: OutcomeKind

    @model_validator(mode="after")
    def selected_outcomes_are_unique(self) -> ScopeBasis:
        if len(self.selected_outcome_ids) != len(set(self.selected_outcome_ids)):
            raise ValueError("scope basis selected outcome ids must be unique")
        return self


class AgentEvidenceFact(EvidenceCertaintyModel):
    fact_id: str
    fact_kind: EvidenceFactKind
    source_quote: str = Field(min_length=1, max_length=20_000)


class AgentTarget(EvidenceCertaintyModel):
    item_id: str
    population: str
    intervention: str
    comparator: str
    outcome: str
    timepoint: str
    estimand: str
    outcome_type: Literal["BINARY"]


class RiskOfBiasSignals(EvidenceCertaintyModel):
    all_contribution_weights_reported: bool
    low_risk_weight_percent: float | None
    unresolved_risk_weight_percent: float | None
    high_risk_weight_percent: float | None


class ImprecisionSignals(EvidenceCertaintyModel):
    null_value: float
    ci_crosses_certainty_threshold: bool
    ci_crosses_null: bool
    relative_ci_ratio: float | None
    relative_effect_change_percent: float | None
    relative_effect_exceeds_30_percent: bool | None
    relative_effect_exceeds_40_percent: bool | None
    ois_alpha: Literal[0.05] = 0.05
    ois_beta: Literal[0.2] = 0.2
    ois_modest_relative_change: Literal[0.2] = 0.2
    ois_required_participants: int | None = Field(default=None, ge=2)
    ois_met: bool | None
    ois_reason: str = Field(min_length=1, max_length=2_000)


class RiskOfBiasAgentInput(EvidenceCertaintyModel):
    grade_rule_version: Literal["CORE_GRADE_2025_V1"]
    domain: Literal["RISK_OF_BIAS"]
    target: AgentTarget
    contributing_studies: tuple[ContributingStudy, ...]
    risk_of_bias_profile: StudyRiskOfBiasProfile
    deterministic_signals: RiskOfBiasSignals
    evidence_facts: tuple[AgentEvidenceFact, ...] = Field(min_length=1)


class InconsistencyAgentInput(EvidenceCertaintyModel):
    grade_rule_version: Literal["CORE_GRADE_2025_V1"]
    domain: Literal["INCONSISTENCY"]
    target: AgentTarget
    synthesis: PublishedSynthesisResult
    single_study: bool
    evidence_facts: tuple[AgentEvidenceFact, ...] = Field(min_length=1)


class IndirectnessAgentInput(EvidenceCertaintyModel):
    grade_rule_version: Literal["CORE_GRADE_2025_V1"]
    domain: Literal["INDIRECTNESS"]
    target: AgentTarget
    protocol_context: ProtocolTargetContext
    evidence_pico: EvidencePicoProfile
    evidence_facts: tuple[AgentEvidenceFact, ...] = Field(min_length=1)


class ImprecisionAgentInput(EvidenceCertaintyModel):
    grade_rule_version: Literal["CORE_GRADE_2025_V1"]
    domain: Literal["IMPRECISION"]
    target: AgentTarget
    certainty_target: CertaintyTarget
    synthesis: PublishedSynthesisResult
    deterministic_signals: ImprecisionSignals
    evidence_facts: tuple[AgentEvidenceFact, ...] = Field(min_length=1)


class PublicationBiasAgentInput(EvidenceCertaintyModel):
    grade_rule_version: Literal["CORE_GRADE_2025_V1"]
    domain: Literal["PUBLICATION_BIAS"]
    target: AgentTarget
    contributing_study_count: int = Field(ge=1)
    evidence_facts: tuple[AgentEvidenceFact, ...] = Field(min_length=1)


class DomainAgentOutput(EvidenceCertaintyModel):
    domain: GradeDomain
    disposition: Literal["ASSESSED", "NOT_ASSESSABLE"]
    judgment: DomainJudgment | None
    rationale: str = Field(min_length=1, max_length=8_000)
    evidence_fact_ids: tuple[str, ...] = Field(max_length=100)
    issues: tuple[str, ...] = Field(max_length=20)

    @model_validator(mode="after")
    def disposition_matches_judgment(self) -> DomainAgentOutput:
        if self.disposition == "ASSESSED" and self.judgment is None:
            raise ValueError("ASSESSED domain requires a judgment")
        if self.disposition == "ASSESSED" and not self.evidence_fact_ids:
            raise ValueError("ASSESSED domain requires source-bound Evidence Facts")
        if self.disposition == "NOT_ASSESSABLE" and self.judgment is not None:
            raise ValueError("NOT_ASSESSABLE domain cannot contain a judgment")
        if len(self.evidence_fact_ids) != len(set(self.evidence_fact_ids)):
            raise ValueError("domain Evidence Fact ids must be unique")
        return self


class OverallDomainSummary(EvidenceCertaintyModel):
    domain: GradeDomain
    judgment: DomainJudgment
    downgrade_lower: int = Field(ge=0, le=2)
    downgrade_upper: int = Field(ge=0, le=2)
    rationale: str
    evidence_fact_ids: tuple[str, ...]


class OverallAgentInput(EvidenceCertaintyModel):
    grade_rule_version: Literal["CORE_GRADE_2025_V1"]
    target: AgentTarget
    initial_certainty: Literal["HIGH"]
    allowed_downgrade_lower: int = Field(ge=0, le=3)
    allowed_downgrade_upper: int = Field(ge=0, le=3)
    domains: tuple[OverallDomainSummary, ...] = Field(min_length=5, max_length=5)


class OverallAgentOutput(EvidenceCertaintyModel):
    disposition: Literal["ASSESSED", "NOT_ASSESSABLE"]
    overall_downgrade_levels: int | None = Field(ge=0, le=3)
    rationale: str = Field(min_length=1, max_length=8_000)
    evidence_fact_ids: tuple[str, ...] = Field(max_length=200)
    issues: tuple[str, ...] = Field(max_length=20)

    @model_validator(mode="after")
    def disposition_matches_downgrade(self) -> OverallAgentOutput:
        if self.disposition == "ASSESSED" and self.overall_downgrade_levels is None:
            raise ValueError("ASSESSED overall view requires overall_downgrade_levels")
        if self.disposition == "ASSESSED" and not self.evidence_fact_ids:
            raise ValueError("ASSESSED overall view requires source-bound Evidence Facts")
        if self.disposition == "NOT_ASSESSABLE" and self.overall_downgrade_levels is not None:
            raise ValueError("NOT_ASSESSABLE overall view cannot contain a downgrade")
        if len(self.evidence_fact_ids) != len(set(self.evidence_fact_ids)):
            raise ValueError("overall Evidence Fact ids must be unique")
        return self


GRADE_DOMAINS: tuple[GradeDomain, ...] = (
    "RISK_OF_BIAS",
    "INCONSISTENCY",
    "INDIRECTNESS",
    "IMPRECISION",
    "PUBLICATION_BIAS",
)


