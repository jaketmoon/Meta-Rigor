from __future__ import annotations

import hashlib
from typing import Annotated, Literal
from pydantic import Field, model_validator
from metarigor.evidence_certainty.models import GRADE_DOMAINS, Certainty, DomainJudgment, EvidenceCertaintyModel, EvidenceFact, GradeDomain, Sha256, SourceDocumentReference, SourceSpan
from metarigor.evidence_certainty.rules import final_certainty, judgment_bounds
class ReviewCandidateAnchor(EvidenceCertaintyModel):
    anchor_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")
    line_number: int = Field(ge=1)
    line_sha256: Sha256
    domain_relevance: tuple[GradeDomain, ...] = Field(min_length=1, max_length=5)


class ReviewEffect(EvidenceCertaintyModel):
    measure: Literal["RR", "OR"]
    estimate: float = Field(gt=0)
    ci_lower: float = Field(gt=0)
    ci_upper: float = Field(gt=0)
    study_count: int = Field(ge=1)
    participant_count: int | None = Field(default=None, ge=1)
    event_count: int | None = Field(default=None, ge=0)
    i2_percent: float | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def estimate_is_inside_interval(self) -> ReviewEffect:
        if not self.ci_lower <= self.estimate <= self.ci_upper:
            raise ValueError("review effect estimate must lie inside its CI")
        return self


class FrozenReviewSource(EvidenceCertaintyModel):
    pmcid: str = Field(pattern=r"^PMC[0-9]+$")
    manifest_path: str = Field(min_length=1, max_length=2_000)
    manifest_sha256: Sha256


class ReviewCaseSourceBindings(EvidenceCertaintyModel):
    selected_analysis: tuple[str, ...] = Field(min_length=1, max_length=5)
    population: tuple[str, ...] = Field(min_length=1, max_length=5)
    intervention: tuple[str, ...] = Field(min_length=1, max_length=5)
    comparator: tuple[str, ...] = Field(min_length=1, max_length=5)
    outcome: tuple[str, ...] = Field(min_length=1, max_length=5)
    timepoint: tuple[str, ...] = Field(min_length=1, max_length=5)
    evidence_design: tuple[str, ...] = Field(min_length=1, max_length=5)
    comparison_kind: tuple[str, ...] = Field(min_length=1, max_length=5)
    synthesis_kind: tuple[str, ...] = Field(min_length=1, max_length=5)
    outcome_kind: tuple[str, ...] = Field(min_length=1, max_length=5)
    effect: tuple[str, ...] = Field(min_length=1, max_length=5)
    effect_tuple_anchor_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")
    effect_tuple_ordinal: int = Field(ge=0, le=9)


class ReviewBenchmarkCase(EvidenceCertaintyModel):
    """Review-level case; role/reason are curation metadata, not semantic source facts."""

    case_id: str = Field(pattern=r"^pmc-[0-9]+-[a-z0-9-]+$")
    selected_analysis_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,159}$")
    review_role: Literal["HEADLINE", "NON_HEADLINE", "SUBGROUP"]
    selection_reason: str = Field(min_length=1, max_length=2_000)
    population: str = Field(min_length=1, max_length=2_000)
    intervention: str = Field(min_length=1, max_length=2_000)
    comparator: str = Field(min_length=1, max_length=2_000)
    outcome: str = Field(min_length=1, max_length=2_000)
    timepoint: str = Field(min_length=1, max_length=500)
    evidence_design: Literal["REVIEW_REPORTED_RCT_EVIDENCE"]
    comparison_kind: Literal["DIRECT_TWO_GROUP"]
    synthesis_kind: Literal["PAIRWISE_META_ANALYSIS"]
    outcome_kind: Literal["BINARY"]
    effect: ReviewEffect
    source: FrozenReviewSource
    anchors: tuple[ReviewCandidateAnchor, ...] = Field(min_length=5, max_length=20)
    source_bindings: ReviewCaseSourceBindings

    @model_validator(mode="after")
    def anchors_are_unique_and_cover_domains(self) -> ReviewBenchmarkCase:
        identities = [item.anchor_id for item in self.anchors]
        lines = [item.line_number for item in self.anchors]
        if len(identities) != len(set(identities)):
            raise ValueError("review candidate anchor ids must be unique")
        if len(lines) != len(set(lines)):
            raise ValueError("review candidate anchor lines must be unique")
        covered = {domain for anchor in self.anchors for domain in anchor.domain_relevance}
        if covered != set(GRADE_DOMAINS):
            raise ValueError("review candidate anchors must cover all five GRADE domains")
        known_anchors = set(identities)
        binding_fields = (
            "selected_analysis",
            "population",
            "intervention",
            "comparator",
            "outcome",
            "timepoint",
            "evidence_design",
            "comparison_kind",
            "synthesis_kind",
            "outcome_kind",
            "effect",
        )
        bound_anchors = {
            anchor_id
            for field_name in binding_fields
            for anchor_id in getattr(self.source_bindings, field_name)
        }
        bound_anchors.add(self.source_bindings.effect_tuple_anchor_id)
        if not bound_anchors <= known_anchors:
            raise ValueError("review case field binding cites an unknown candidate anchor")
        if self.source_bindings.effect_tuple_anchor_id not in self.source_bindings.effect:
            raise ValueError("selected effect tuple anchor must be an effect binding")
        if self.source.pmcid.removeprefix("PMC") not in self.case_id:
            raise ValueError("review case id must bind its PMCID")
        return self


class BenchmarkV2Catalog(EvidenceCertaintyModel):
    schema_version: Literal["2.1.0"]
    benchmark_id: Literal["metarigor-evidence-certainty-benchmark-v2-review"]
    grade_rule_version: Literal["CORE_GRADE_2025_V1"]
    scope: Literal["REVIEW_LEVEL_DIRECT_TWO_GROUP_BINARY_RCT"]
    inherited_v1_catalog_sha256: Sha256
    inherited_v1_scored_case_ids: tuple[str, ...] = Field(min_length=7, max_length=7)
    retained_v1_abstention_case_ids: tuple[str, ...] = Field(min_length=3, max_length=3)
    review_cases: tuple[ReviewBenchmarkCase, ...] = Field(min_length=13, max_length=13)

    @model_validator(mode="after")
    def denominators_are_frozen(self) -> BenchmarkV2Catalog:
        inherited = set(self.inherited_v1_scored_case_ids)
        abstained = set(self.retained_v1_abstention_case_ids)
        review = {item.case_id for item in self.review_cases}
        if len(review) != 13 or inherited & abstained or inherited & review or abstained & review:
            raise ValueError("EC V2 case identities must be unique")
        return self

    @property
    def scored_case_ids(self) -> tuple[str, ...]:
        return self.inherited_v1_scored_case_ids + tuple(item.case_id for item in self.review_cases)


class ReviewGoldDomain(EvidenceCertaintyModel):
    domain: GradeDomain
    judgment: DomainJudgment
    rationale: str = Field(min_length=1, max_length=8_000)
    evidence_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=20)


class ReviewGoldCase(EvidenceCertaintyModel):
    case_id: str
    selected_analysis_id: str
    domains: tuple[ReviewGoldDomain, ...] = Field(min_length=5, max_length=5)
    overall_downgrade_levels: int = Field(ge=0, le=3)
    final_certainty: Certainty
    overall_rationale: str = Field(min_length=1, max_length=8_000)
    curator_confidence: Literal["HIGH", "MODERATE", "LOW"]

    @model_validator(mode="after")
    def result_respects_core_grade_bounds(self) -> ReviewGoldCase:
        domains = [item.domain for item in self.domains]
        if len(domains) != len(set(domains)) or set(domains) != set(GRADE_DOMAINS):
            raise ValueError("review gold must contain the five unique frozen domains")
        lower = min(3, max(judgment_bounds(item.judgment)[0] for item in self.domains))
        upper = min(3, sum(judgment_bounds(item.judgment)[1] for item in self.domains))
        if not lower <= self.overall_downgrade_levels <= upper:
            raise ValueError("review gold overall downgrade is outside domain bounds")
        if self.final_certainty != final_certainty(self.overall_downgrade_levels):
            raise ValueError("review gold final certainty differs from downgrade level")
        return self


class BenchmarkV2Gold(EvidenceCertaintyModel):
    schema_version: Literal["2.1.0"]
    benchmark_id: Literal["metarigor-evidence-certainty-benchmark-v2-review"]
    candidate_catalog_sha256: Sha256
    grade_rule_version: Literal["CORE_GRADE_2025_V1"]
    reference_maturity: Literal["CURATOR_ADJUDICATED_NOT_CLINICAL_EXPERT_VALIDATED"]
    external_comparator_blinded_during_adjudication: Literal[True]
    inherited_v1_gold_sha256: Sha256
    cases: tuple[ReviewGoldCase, ...] = Field(min_length=13, max_length=13)

    @model_validator(mode="after")
    def thirteen_cases_are_unique(self) -> BenchmarkV2Gold:
        identities = [item.case_id for item in self.cases]
        if len(identities) != len(set(identities)):
            raise ValueError("EC V2 review gold case ids must be unique")
        return self


class ExternalComparatorDocument(EvidenceCertaintyModel):
    case_id: str
    pmcid: str
    path: str
    sha256: Sha256
    content: str = Field(min_length=1)


class ReviewCandidateDocument(EvidenceCertaintyModel):
    case_id: str
    pmcid: str
    path: str
    sha256: Sha256
    content: str = Field(min_length=1)


class ReviewFigureEvidence(EvidenceCertaintyModel):
    """Manually verified Review-image transcription for the V2 benchmark only; not a production Source Span."""

    evidence_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")
    representation_kind: Literal["CURATOR_VERIFIED_REVIEW_FIGURE_TRANSCRIPTION"]
    source_pmcid: str = Field(pattern=r"^PMC[0-9]+$")
    figure_label: str = Field(min_length=1, max_length=160)
    artifact_path: str = Field(min_length=1, max_length=2_000)
    artifact_sha256: Sha256
    source_url: str = Field(
        pattern=(
            r"^https://(?:cdn\.ncbi\.nlm\.nih\.gov/pmc/blobs/|"
            r"pmc-oa-opendata\.s3\.amazonaws\.com/)"
        )
    )
    source_document_path: str | None = Field(default=None, min_length=1, max_length=2_000)
    source_document_sha256: Sha256 | None = None
    derivation_locator: str | None = Field(default=None, min_length=1, max_length=1_000)
    transcription: str = Field(min_length=1, max_length=20_000)

    @model_validator(mode="after")
    def derived_artifact_has_complete_source(self) -> ReviewFigureEvidence:
        source_pair = (self.source_document_path, self.source_document_sha256)
        if (source_pair[0] is None) != (source_pair[1] is None):
            raise ValueError("review figure source document path and SHA-256 must be paired")
        if (self.derivation_locator is None) != (self.source_document_path is None):
            raise ValueError("derived review figure requires a source document and locator")
        return self


class ReviewTextEvidence(EvidenceCertaintyModel):
    """Exact single-line excerpt from a frozen Review candidate."""

    evidence_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,63}$")
    representation_kind: Literal["FROZEN_REVIEW_TEXT_EXCERPT"]
    source_pmcid: str = Field(pattern=r"^PMC[0-9]+$")
    candidate_path: str = Field(min_length=1, max_length=2_000)
    candidate_sha256: Sha256
    line_number: int = Field(ge=1)
    line_sha256: Sha256
    quote: str = Field(min_length=1, max_length=20_000)


ReviewSupplementalEvidence = Annotated[
    ReviewFigureEvidence | ReviewTextEvidence,
    Field(discriminator="representation_kind"),
]


ReviewSufficiencyBasis = Literal[
    "COMPLETE_REVIEW_FOREST",
    "REVIEW_DIRECTION_MAGNITUDE_CLOSURE",
    "SOURCE_IDENTITY_CONFLICT",
    "MISSING_SELECTED_SYNTHESIS_CONTRIBUTOR_IDENTITY",
    "REVIEW_ROB_WITH_CONTRIBUTION_CONTEXT",
    "OUTCOME_SPECIFIC_LOW_RISK_SENSITIVITY",
    "MISSING_OUTCOME_RELEVANT_ROB",
    "REVIEW_SEARCH_AND_SMALL_STUDY_ASSESSMENT",
]


class ReviewSourceResolution(EvidenceCertaintyModel):
    """Pin the governing version of conflicting sources before model invocation."""

    resolution_kind: Literal["USER_AUTHORIZED_GOVERNING_SOURCE_SELECTION"]
    authority: Literal["USER_INSTRUCTION_2026_09_01"]
    decision_id: Literal["ec-v2-pmc134-figure2-governing-version-20260901"]
    authorization_actor: Literal["USER"]
    selection_actor: Literal["USER"]
    decision_text: str = Field(min_length=1, max_length=1_000)
    decision_text_sha256: Sha256
    source_pmcid: str = Field(pattern=r"^PMC[0-9]+$")
    selected_version: str = Field(min_length=1, max_length=1_000)
    selected_artifact_path: str = Field(min_length=1, max_length=2_000)
    selected_artifact_sha256: Sha256
    selected_source_url: str = Field(
        pattern=(
            r"^https://(?:cdn\.ncbi\.nlm\.nih\.gov/pmc/blobs/|"
            r"pmc-oa-opendata\.s3\.amazonaws\.com/)"
        )
    )
    excluded_version: str = Field(min_length=1, max_length=1_000)
    excluded_artifact_path: str = Field(min_length=1, max_length=2_000)
    excluded_artifact_sha256: Sha256
    excluded_source_document_path: str = Field(min_length=1, max_length=2_000)
    excluded_source_document_sha256: Sha256
    selected_study_count: Literal[15]
    selected_intervention_total: Literal[1023]
    selected_control_total: Literal[1050]
    excluded_study_count: Literal[14]
    excluded_intervention_total: Literal[988]
    excluded_control_total: Literal[1008]
    rationale: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def decision_text_is_hash_bound(self) -> ReviewSourceResolution:
        if hashlib.sha256(self.decision_text.encode("utf-8")).hexdigest() != (
            self.decision_text_sha256
        ):
            raise ValueError("review source-resolution decision text hash changed")
        if self.selected_version == self.excluded_version:
            raise ValueError("selected and excluded source versions must differ")
        return self


class ReviewDomainSufficiency(EvidenceCertaintyModel):
    domain: Literal["RISK_OF_BIAS", "INCONSISTENCY", "PUBLICATION_BIAS"]
    disposition: Literal["ASSESSABLE", "NOT_ASSESSABLE"]
    basis: ReviewSufficiencyBasis
    missing_facts: tuple[str, ...] = Field(max_length=10)
    issue: str | None = Field(default=None, max_length=4_000)
    supplemental_evidence: tuple[ReviewSupplementalEvidence, ...] = Field(max_length=5)
    source_resolution: ReviewSourceResolution | None = None

    @model_validator(mode="after")
    def disposition_is_honest(self) -> ReviewDomainSufficiency:
        if self.disposition == "ASSESSABLE" and self.missing_facts:
            raise ValueError("assessable review domain cannot list required missing facts")
        if self.disposition == "NOT_ASSESSABLE" and not self.missing_facts:
            raise ValueError("review domain abstention must list required missing facts")
        allowed_bases = {
            "RISK_OF_BIAS": {
                "REVIEW_ROB_WITH_CONTRIBUTION_CONTEXT",
                "OUTCOME_SPECIFIC_LOW_RISK_SENSITIVITY",
                "MISSING_OUTCOME_RELEVANT_ROB",
            },
            "INCONSISTENCY": {
                "COMPLETE_REVIEW_FOREST",
                "REVIEW_DIRECTION_MAGNITUDE_CLOSURE",
                "SOURCE_IDENTITY_CONFLICT",
                "MISSING_SELECTED_SYNTHESIS_CONTRIBUTOR_IDENTITY",
            },
            "PUBLICATION_BIAS": {"REVIEW_SEARCH_AND_SMALL_STUDY_ASSESSMENT"},
        }
        if self.basis not in allowed_bases[self.domain]:
            raise ValueError("review sufficiency basis is invalid for domain")
        abstention_bases = {
            "SOURCE_IDENTITY_CONFLICT",
            "MISSING_SELECTED_SYNTHESIS_CONTRIBUTOR_IDENTITY",
            "MISSING_OUTCOME_RELEVANT_ROB",
        }
        if (self.disposition == "NOT_ASSESSABLE") != (self.basis in abstention_bases):
            raise ValueError("review sufficiency basis contradicts disposition")
        if self.source_resolution is not None and self.domain != "INCONSISTENCY":
            raise ValueError("source resolution is only valid for inconsistency")
        return self


class ReviewCaseSufficiency(EvidenceCertaintyModel):
    case_id: str
    domains: tuple[ReviewDomainSufficiency, ...] = Field(min_length=2, max_length=3)

    @model_validator(mode="after")
    def two_review_domains_are_unique(self) -> ReviewCaseSufficiency:
        domains = tuple(item.domain for item in self.domains)
        canonical = ("RISK_OF_BIAS", "INCONSISTENCY", "PUBLICATION_BIAS")
        if domains[:2] != canonical[:2] or domains != tuple(
            domain for domain in canonical if domain in domains
        ):
            raise ValueError("review sufficiency domains must use canonical adapted order")
        return self


class ReviewLevelSufficiencyContract(EvidenceCertaintyModel):
    schema_version: Literal["3.0.0"]
    contract_id: Literal["EC_V2_REVIEW_LEVEL_SUFFICIENCY_V3"]
    governing_grade_rule_version: Literal["CORE_GRADE_2025_V1"]
    scope: Literal["V2_PMC_OA_REVIEW_LEVEL_ONLY"]
    cases: tuple[ReviewCaseSufficiency, ...] = Field(min_length=13, max_length=13)

    @model_validator(mode="after")
    def thirteen_cases_are_unique(self) -> ReviewLevelSufficiencyContract:
        identities = [item.case_id for item in self.cases]
        if len(identities) != len(set(identities)):
            raise ValueError("review sufficiency case ids must be unique")
        return self


class BenchmarkV2EvidenceAnchorSidecar(EvidenceCertaintyModel):
    """Candidate evidence-to-SourceSpan mapping closed by the program before model invocation."""

    evidence_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,159}$")
    domain_relevance: tuple[GradeDomain, ...] = Field(min_length=1, max_length=5)
    statement: str = Field(min_length=1, max_length=20_000)
    evidence_fact: EvidenceFact | None = None
    source_lineage: Literal[
        "PRIMARY_SOURCE_CLOSED",
        "REVIEW_DOCUMENT_ONLY",
        "SOURCE_LINEAGE_GAP",
    ]
    source_path: str | None = Field(default=None, min_length=1, max_length=10_000)
    source_sha256: Sha256 | None = None
    view_path: str | None = Field(default=None, min_length=1, max_length=10_000)
    view_sha256: Sha256 | None = None
    source_span: SourceSpan | None = None
    raw_source_provenance: tuple[SourceDocumentReference, ...] = Field(default=(), max_length=20)
    issue_codes: tuple[str, ...] = Field(default=(), max_length=20)
    issue_detail: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def paths_and_hashes_match_source_span(self) -> BenchmarkV2EvidenceAnchorSidecar:
        if len(self.domain_relevance) != len(set(self.domain_relevance)):
            raise ValueError("evidence sidecar domain relevance must be unique")
        closure_fields = (
            self.evidence_fact,
            self.source_path,
            self.source_sha256,
            self.view_path,
            self.view_sha256,
            self.source_span,
        )
        if self.source_lineage == "SOURCE_LINEAGE_GAP":
            if any(item is not None for item in closure_fields):
                raise ValueError("lineage-gap sidecar cannot expose a closed Source Span")
            if not self.issue_codes or self.issue_detail is None:
                raise ValueError("lineage-gap sidecar requires an explicit Issue")
            if self.raw_source_provenance:
                raise ValueError("lineage-gap sidecar cannot claim raw provenance")
            return self
        if any(item is None for item in closure_fields):
            raise ValueError("closed evidence sidecar requires EvidenceFact and Source Span")
        evidence_fact = self.evidence_fact
        source_span = self.source_span
        if evidence_fact is None or source_span is None:
            raise ValueError("closed evidence sidecar requires typed closure fields")
        if (
            evidence_fact.fact_id != self.evidence_id
            or evidence_fact.statement != self.statement
            or evidence_fact.domain_relevance != self.domain_relevance
            or evidence_fact.source_span != source_span
        ):
            raise ValueError("evidence sidecar differs from its formal EvidenceFact")
        if (
            source_span.input_path != self.view_path
            or source_span.document_sha256 != self.view_sha256
        ):
            raise ValueError("evidence sidecar view binding differs from Source Span")
        if source_span.derivation_sources != self.raw_source_provenance:
            raise ValueError("evidence sidecar raw provenance differs from Source Span")
        source_pairs = {
            (item.input_path, item.document_sha256) for item in self.raw_source_provenance
        }
        if source_span.source_representation == "RAW_SOURCE":
            source_pairs.add((self.view_path, self.view_sha256))
        if (self.source_path, self.source_sha256) not in source_pairs:
            raise ValueError("evidence sidecar source is not a terminal raw source")
        if self.issue_codes or self.issue_detail is not None:
            raise ValueError("closed evidence sidecar cannot carry lineage-gap Issues")
        return self


class BenchmarkV2EvidenceClaim(EvidenceCertaintyModel):
    """Shared answer-level evidence-claim representation for MR and blinded external candidates."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    claim_id: str = Field(min_length=1, max_length=500)
    case_id: str = Field(min_length=1, max_length=200)
    domain: GradeDomain
    claim_text: str = Field(min_length=1, max_length=20_000)
    evidence_id: str | None = Field(default=None, max_length=200)
    support_status: Literal["SUPPORTED", "UNSUPPORTED", "AMBIGUOUS"]
    source_lineage: Literal[
        "PRIMARY_SOURCE_CLOSED",
        "REVIEW_DOCUMENT_ONLY",
        "CANDIDATE_PACKAGE_ONLY",
        "SOURCE_LINEAGE_GAP",
    ]
    source_path: str | None = Field(default=None, max_length=10_000)
    source_sha256: Sha256 | None = None
    view_path: str | None = Field(default=None, max_length=10_000)
    view_sha256: Sha256 | None = None
    source_span: SourceSpan | None = None
    issue_codes: tuple[str, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def supported_claim_has_exact_source(self) -> BenchmarkV2EvidenceClaim:
        source_fields = (
            self.source_path,
            self.source_sha256,
            self.view_path,
            self.view_sha256,
            self.source_span,
        )
        if self.support_status == "SUPPORTED":
            if self.source_lineage == "SOURCE_LINEAGE_GAP":
                raise ValueError("supported evidence claim cannot be a source-lineage gap")
            if self.evidence_id is None or any(item is None for item in source_fields):
                raise ValueError("supported evidence claim requires a complete Source Span")
            source_span = self.source_span
            if source_span is None:
                raise ValueError("supported evidence claim requires a Source Span")
            if (
                source_span.input_path != self.view_path
                or source_span.document_sha256 != self.view_sha256
            ):
                raise ValueError("evidence claim view binding differs from Source Span")
            if self.issue_codes:
                raise ValueError("supported evidence claim cannot carry verifier issues")
        elif self.source_span is not None:
            raise ValueError("unsupported evidence claim cannot expose a valid Source Span")
        elif not self.issue_codes:
            raise ValueError("unsupported evidence claim requires an explicit issue")
        return self


class BenchmarkV2PredictionDomain(EvidenceCertaintyModel):
    domain: GradeDomain
    judgment: DomainJudgment
    rationale: str | None = Field(default=None, max_length=8_000)
    evidence_anchor_ids: tuple[str, ...] = Field(default=(), max_length=100)
    evidence_claims: tuple[BenchmarkV2EvidenceClaim, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def claims_match_selected_evidence(self) -> BenchmarkV2PredictionDomain:
        if any(claim.domain != self.domain for claim in self.evidence_claims):
            raise ValueError("evidence claim domain differs from its parent domain")
        claim_ids = tuple(claim.evidence_id for claim in self.evidence_claims)
        if (self.evidence_anchor_ids or self.evidence_claims) and claim_ids != tuple(
            self.evidence_anchor_ids
        ):
            raise ValueError("evidence claims must exactly match selected evidence IDs")
        return self


class BenchmarkV2EvidenceVerificationIssue(EvidenceCertaintyModel):
    issue_code: Literal[
        "UNKNOWN_EVIDENCE_ID",
        "DUPLICATE_EVIDENCE_ID",
        "EVIDENCE_DOMAIN_MISMATCH",
        "DOMAIN_SOURCE_SPAN_MISSING",
        "SOURCE_FILE_MISSING",
        "SOURCE_SHA256_MISMATCH",
        "VIEW_FILE_MISSING",
        "VIEW_SHA256_MISMATCH",
        "SOURCE_SPAN_OFFSET_MISMATCH",
        "QUOTE_SHA256_MISMATCH",
        "SOURCE_IDENTITY_MISMATCH",
        "SOURCE_LINEAGE_GAP",
        "OVERALL_EVIDENCE_OUTSIDE_DOMAIN_UNION",
        "OVERALL_SOURCE_SPAN_MISMATCH",
    ]
    case_id: str
    domain: GradeDomain | None = None
    evidence_id: str | None = None
    detail: str = Field(min_length=1, max_length=4_000)


class BenchmarkV2EvidenceVerification(EvidenceCertaintyModel):
    case_id: str
    verdict: Literal["PASSED", "FAILED"]
    issues: tuple[BenchmarkV2EvidenceVerificationIssue, ...] = Field(max_length=1_000)

    @model_validator(mode="after")
    def verdict_matches_issues(self) -> BenchmarkV2EvidenceVerification:
        if (self.verdict == "PASSED") != (not self.issues):
            raise ValueError("evidence verification verdict differs from Issue presence")
        return self


class BenchmarkV2PredictionCase(EvidenceCertaintyModel):
    case_id: str
    domains: tuple[BenchmarkV2PredictionDomain, ...] = Field(min_length=5, max_length=5)
    overall_downgrade_levels: int = Field(ge=0, le=3)
    final_certainty: Certainty
    overall_rationale: str | None = Field(default=None, max_length=8_000)
    overall_evidence_anchor_ids: tuple[str, ...] = Field(default=(), max_length=500)
    overall_source_spans: tuple[SourceSpan, ...] = Field(default=(), max_length=500)
    evidence_verification: BenchmarkV2EvidenceVerification | None = None

    @model_validator(mode="after")
    def five_domains_are_unique(self) -> BenchmarkV2PredictionCase:
        domains = [item.domain for item in self.domains]
        if len(domains) != len(set(domains)) or set(domains) != set(GRADE_DOMAINS):
            raise ValueError("prediction must contain the five unique frozen domains")
        lower = min(3, max(judgment_bounds(item.judgment)[0] for item in self.domains))
        upper = min(3, sum(judgment_bounds(item.judgment)[1] for item in self.domains))
        if not lower <= self.overall_downgrade_levels <= upper:
            raise ValueError("prediction overall downgrade is outside domain bounds")
        if self.final_certainty != final_certainty(self.overall_downgrade_levels):
            raise ValueError("prediction final certainty differs from downgrade level")
        if self.evidence_verification is not None:
            if self.evidence_verification.case_id != self.case_id:
                raise ValueError("evidence verification case differs from prediction case")
            if any(
                claim.case_id != self.case_id
                for domain in self.domains
                for claim in domain.evidence_claims
            ):
                raise ValueError("evidence claim case differs from prediction case")
            domain_union = tuple(
                dict.fromkeys(
                    claim.evidence_id
                    for domain in self.domains
                    for claim in domain.evidence_claims
                    if claim.support_status == "SUPPORTED" and claim.evidence_id is not None
                )
            )
            if self.overall_evidence_anchor_ids != domain_union:
                raise ValueError("prediction overall evidence differs from the domain union")
            domain_spans = tuple(
                dict.fromkeys(
                    claim.source_span
                    for domain in self.domains
                    for claim in domain.evidence_claims
                    if claim.support_status == "SUPPORTED" and claim.source_span is not None
                )
            )
            if self.overall_source_spans != domain_spans:
                raise ValueError("prediction overall Source Spans differ from domain evidence")
            if self.evidence_verification.verdict == "PASSED" and any(
                not any(
                    claim.support_status == "SUPPORTED" and claim.source_span is not None
                    for claim in domain.evidence_claims
                )
                for domain in self.domains
            ):
                raise ValueError(
                    "passed evidence verification requires a valid Source Span for every domain"
                )
        return self


class BenchmarkV2PredictionSet(EvidenceCertaintyModel):
    schema_version: Literal["1.0.0", "2.0.0"] = "1.0.0"
    benchmark_id: Literal["metarigor-evidence-certainty-benchmark-v2-review"]
    cases: tuple[BenchmarkV2PredictionCase, ...] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def answer_evidence_v2_is_explicit(self) -> BenchmarkV2PredictionSet:
        if self.schema_version == "2.0.0":
            case_ids = tuple(case.case_id for case in self.cases)
            if len(set(case_ids)) != len(case_ids):
                raise ValueError("V2 prediction set requires twenty unique case IDs")
            if any(
                case.evidence_verification is None
                or case.overall_rationale is None
                or any(domain.rationale is None for domain in case.domains)
                for case in self.cases
            ):
                raise ValueError("V2 prediction set requires explicit answer-evidence verification")
        return self


class V2CandidateSeal(EvidenceCertaintyModel):
    catalog_sha256: Sha256
    prediction_set_sha256: Sha256
    prediction_set: BenchmarkV2PredictionSet

    @model_validator(mode="after")
    def prediction_hash_is_closed(self) -> V2CandidateSeal:
        if self.prediction_set.canonical_sha256 != self.prediction_set_sha256:
            raise ValueError("candidate seal prediction hash changed")
        return self


class BenchmarkV2EvidenceMetrics(EvidenceCertaintyModel):
    answer_evidence_coverage_numerator: int = Field(ge=0, le=100)
    answer_evidence_coverage_denominator: int = Field(ge=0, le=100)
    source_span_closure_numerator: int = Field(ge=0)
    source_span_closure_denominator: int = Field(ge=0)
    reference_anchor_recall_numerator: int = Field(ge=0, le=141)
    reference_anchor_recall_denominator: Literal[141] = 141
    raw_source_closure_numerator: int = Field(ge=0)
    raw_source_closure_denominator: int = Field(ge=0)
    unsupported_evidence_count: int = Field(ge=0)
    case_evidence_complete_numerator: int = Field(ge=0, le=20)
    case_evidence_complete_denominator: Literal[20] = 20
    primary_source_closed_selected_span_count: int = Field(ge=0)
    review_document_only_selected_span_count: int = Field(ge=0)

