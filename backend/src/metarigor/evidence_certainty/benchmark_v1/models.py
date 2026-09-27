from __future__ import annotations

import json
from typing import Literal
from pydantic import Field, model_validator
from metarigor.evidence_certainty.models import Certainty, CertaintyTarget, DomainJudgment, EvidenceCertaintyModel, EvidenceFactKind, EvidencePicoProfile, GradeDomain, PublishedSynthesisResult, ScopeBasis, ScopeBasisField, Sha256, SourceRepresentation, StudyDesignSupportField, StudyRiskOfBiasProfile
from metarigor.evidence_certainty.rules import final_certainty, judgment_bounds
class BenchmarkSourceDocument(EvidenceCertaintyModel):
    document_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    path: str = Field(min_length=1, max_length=2_000)
    sha256: Sha256
    source_representation: SourceRepresentation


class BenchmarkSourceLineage(EvidenceCertaintyModel):
    document_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    document_sha256: Sha256
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
    def output_fields_are_unique(self) -> BenchmarkSourceLineage:
        if len(self.output_fields) != len(set(self.output_fields)):
            raise ValueError("benchmark source lineage output_fields must be unique")
        if (
            self.derivation == "DETERMINISTIC_PDF_TEXT_EXTRACTION"
            and self.locator_kind != "PDF_PAGES"
        ):
            raise ValueError("deterministic PDF text extraction requires PDF_PAGES")
        return self


class BenchmarkEvidenceFact(EvidenceCertaintyModel):
    fact_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,159}$")
    fact_kind: EvidenceFactKind
    statement: str = Field(min_length=1, max_length=8_000)
    domain_relevance: tuple[GradeDomain, ...] = Field(min_length=1, max_length=5)
    document_id: str
    quote: str = Field(min_length=1, max_length=20_000)


class BenchmarkScopeBasisEvidence(EvidenceCertaintyModel):
    fields: tuple[ScopeBasisField, ...] = Field(min_length=1, max_length=5)
    document_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    quote: str = Field(min_length=1, max_length=20_000)

    @model_validator(mode="after")
    def fields_are_unique(self) -> BenchmarkScopeBasisEvidence:
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("benchmark scope basis fields must be unique")
        return self


class BenchmarkStudyDesignEvidence(EvidenceCertaintyModel):
    report_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,159}$")
    document_role: Literal["PRIMARY_REPORT", "REGISTRY_RECORD"]
    fields: tuple[StudyDesignSupportField, ...] = Field(min_length=1, max_length=3)
    document_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    quote: str = Field(min_length=1, max_length=20_000)

    @model_validator(mode="after")
    def fields_are_unique(self) -> BenchmarkStudyDesignEvidence:
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("benchmark Study design evidence fields must be unique")
        return self


class BenchmarkStudyDesignBasis(EvidenceCertaintyModel):
    study_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    inference: Literal[
        "EXPLICIT_PARALLEL",
        "INDIVIDUAL_TWO_FIXED_ARMS_SINGLE_EPISODE",
    ]
    evidence: tuple[BenchmarkStudyDesignEvidence, ...] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def evidence_closes_design(self) -> BenchmarkStudyDesignBasis:
        fields = [field for evidence in self.evidence for field in evidence.fields]
        if len(fields) != len(set(fields)) or set(fields) != {
            "PARTICIPANT_RANDOMIZATION",
            "EXACTLY_TWO_FIXED_ARMS",
            "PARALLEL_ASSIGNMENT_CONTEXT",
        }:
            raise ValueError("benchmark Study design basis is not closed")
        if not any(item.document_role == "PRIMARY_REPORT" for item in self.evidence):
            raise ValueError("benchmark Study design basis requires a primary Report binding")
        return self


class BenchmarkCaseDefinition(EvidenceCertaintyModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,159}$")
    legacy_split: Literal["development", "regression"]
    legacy_case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,239}$")
    selected_analysis_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,159}$")
    selection_reason: str = Field(min_length=1, max_length=4_000)
    source_closure: Literal[
        "CLOSED",
        "CLOSED_RECONCILED_SOURCE_CONFLICT",
        "REFERENCE_CONFLICT",
        "UNSUPPORTED_SYNTHESIS_METHOD",
        "SOURCE_GATED",
    ]
    source_closure_rationale: str = Field(min_length=1, max_length=4_000)
    source_closure_document_id: str | None
    source_closure_quote: str | None = Field(max_length=20_000)
    scope_basis: ScopeBasis
    scope_basis_evidence: tuple[BenchmarkScopeBasisEvidence, ...] = Field(
        min_length=1, max_length=10
    )
    study_design_bases: tuple[BenchmarkStudyDesignBasis, ...] = Field(default=(), max_length=10_000)
    protocol_asset_path: str = Field(min_length=1, max_length=2_000)
    protocol_asset_sha256: Sha256
    source_documents: tuple[BenchmarkSourceDocument, ...] = Field(min_length=1, max_length=50)
    source_derivations: dict[str, tuple[BenchmarkSourceLineage, ...]] = Field(default_factory=dict)
    contributing_study_ids: tuple[str, ...] = Field(min_length=1)
    study_risk_of_bias: dict[str, Literal["LOW", "SOME_CONCERNS", "HIGH", "NOT_ASSESSED"]]
    contribution_weight_percent: dict[str, float] = Field(default_factory=dict)
    risk_of_bias_profile: StudyRiskOfBiasProfile | None = None
    evidence_pico: EvidencePicoProfile | None = None
    synthesis: PublishedSynthesisResult
    certainty_target: CertaintyTarget
    evidence_facts: tuple[BenchmarkEvidenceFact, ...] = Field(min_length=5)

    @model_validator(mode="after")
    def closure_is_unique(self) -> BenchmarkCaseDefinition:
        documents = [item.document_id for item in self.source_documents]
        facts = [item.fact_id for item in self.evidence_facts]
        if len(documents) != len(set(documents)):
            raise ValueError("benchmark Source Document ids must be unique")
        if len(facts) != len(set(facts)):
            raise ValueError("benchmark Evidence Fact ids must be unique")
        known_documents = set(documents)
        if any(item.document_id not in known_documents for item in self.evidence_facts):
            raise ValueError("benchmark Evidence Fact references an unknown Source Document")
        if any(item.document_id not in known_documents for item in self.scope_basis_evidence):
            raise ValueError("benchmark scope basis references an unknown Source Document")
        if any(
            evidence.document_id not in known_documents
            for basis in self.study_design_bases
            for evidence in basis.evidence
        ):
            raise ValueError("benchmark Study design basis references an unknown Source Document")
        document_hashes = {item.document_id: item.sha256 for item in self.source_documents}
        if any(document_id not in known_documents for document_id in self.source_derivations):
            raise ValueError("benchmark source derivation key is an unknown Source Document")
        if any(not sources for sources in self.source_derivations.values()):
            raise ValueError("benchmark source derivation cannot be empty")
        lineage = tuple(
            source for sources in self.source_derivations.values() for source in sources
        )
        if any(source.document_id not in known_documents for source in lineage):
            raise ValueError("benchmark source lineage references an unknown Source Document")
        if any(document_hashes[source.document_id] != source.document_sha256 for source in lineage):
            raise ValueError("benchmark source lineage SHA-256 differs from Source Document")
        if any(source.document_id in self.source_derivations for source in lineage):
            raise ValueError("benchmark source lineage must terminate at a raw Source Document")
        representations = {
            document.document_id: document.source_representation
            for document in self.source_documents
        }
        curated_documents = {
            document_id
            for document_id, representation in representations.items()
            if representation == "CURATED_DERIVATION"
        }
        if any(representations[source.document_id] != "RAW_SOURCE" for source in lineage):
            raise ValueError("benchmark source lineage must terminate at raw Source Documents")
        used_documents = {fact.document_id for fact in self.evidence_facts}
        used_documents.update(item.document_id for item in self.scope_basis_evidence)
        used_documents.update(
            evidence.document_id for basis in self.study_design_bases for evidence in basis.evidence
        )
        if self.source_closure_document_id is not None:
            used_documents.add(self.source_closure_document_id)
        derived_documents = set(self.source_derivations)
        if not derived_documents <= used_documents:
            raise ValueError("benchmark source derivation output is unused")
        required_documents = used_documents | {source.document_id for source in lineage}
        if required_documents != known_documents:
            raise ValueError("benchmark Source Document set contains unused inputs")
        if any(
            document_id not in self.source_derivations
            for document_id in used_documents & curated_documents
        ):
            raise ValueError("curated benchmark Evidence Fact requires upstream source lineage")
        if derived_documents != curated_documents:
            raise ValueError("benchmark curated Source Documents must match derivation keys")
        output_fields = {
            document_id: {field for source in sources for field in source.output_fields}
            for document_id, sources in self.source_derivations.items()
        }
        design_outputs: dict[str, set[str]] = {}
        for basis in self.study_design_bases:
            for evidence in basis.evidence:
                if representations[evidence.document_id] != "CURATED_DERIVATION":
                    continue
                design_outputs.setdefault(evidence.document_id, set()).update(
                    f"study_design_bases.{basis.study_id}.{field}" for field in evidence.fields
                )
        if any(
            output_fields.get(document_id) != required
            for document_id, required in design_outputs.items()
        ):
            raise ValueError("benchmark Study design derivation fields are incomplete")
        required_outputs = {
            "ec-risk-of-bias-profiles": {
                "risk_of_bias_profile.study_judgments",
                "risk_of_bias_profile.contribution_weight_percent",
                "risk_of_bias_profile.observations",
            },
            "ec-evidence-pico-profiles": {
                "evidence_pico.population",
                "evidence_pico.intervention",
                "evidence_pico.comparator",
                "evidence_pico.outcome",
                "evidence_pico.timepoint",
                "evidence_pico.setting",
                "evidence_pico.estimand",
                "evidence_pico.notable_variations",
            },
            "ec-synthesis-transcriptions": {
                "synthesis.effect_measure",
                "synthesis.estimate",
                "synthesis.ci_lower",
                "synthesis.ci_upper",
                "synthesis.confidence_level",
                "synthesis.study_count",
                "synthesis.participant_count",
                "synthesis.event_count",
                "synthesis.control_event_count",
                "synthesis.control_participant_count",
                "synthesis.i2_percent",
                "synthesis.prediction_interval_lower",
                "synthesis.prediction_interval_upper",
                "synthesis.analysis_method",
            },
        }
        for document_id, required in required_outputs.items():
            if document_id in used_documents and output_fields.get(document_id) != required:
                raise ValueError(
                    f"benchmark source lineage does not cover {document_id} fields exactly"
                )
        for fact in self.evidence_facts:
            if fact.fact_id == "study-effects" and fact.document_id in self.source_derivations:
                if output_fields[fact.document_id] != {"study_effects"}:
                    raise ValueError("study-effect transcription lineage is incomplete")
        if len(self.contributing_study_ids) != len(set(self.contributing_study_ids)):
            raise ValueError("benchmark contributing Study ids must be unique")
        if set(self.study_risk_of_bias) != set(self.contributing_study_ids):
            raise ValueError("benchmark RoB map must close contributing Studies")
        if self.contribution_weight_percent and set(self.contribution_weight_percent) != set(
            self.contributing_study_ids
        ):
            raise ValueError("benchmark contribution weights must be complete or absent")
        if any(not 0 <= value <= 100 for value in self.contribution_weight_percent.values()):
            raise ValueError("benchmark contribution weights must be percentages")
        if (
            self.contribution_weight_percent
            and abs(sum(self.contribution_weight_percent.values()) - 100) > 0.05
        ):
            raise ValueError("benchmark contribution weights must sum to 100")
        if self.synthesis.study_count != len(self.contributing_study_ids):
            raise ValueError("benchmark Study closure must match synthesis study_count")
        score_eligible_without_design = (
            self.scope_basis.comparison_kind == "DIRECT_TWO_ARM"
            and self.scope_basis.synthesis_kind in {"PAIRWISE_META_ANALYSIS", "SINGLE_STUDY"}
            and self.scope_basis.outcome_kind == "BINARY"
            and self.source_closure in {"CLOSED", "CLOSED_RECONCILED_SOURCE_CONFLICT"}
        )
        design_ids = [basis.study_id for basis in self.study_design_bases]
        if len(design_ids) != len(set(design_ids)):
            raise ValueError("benchmark Study design basis ids must be unique")
        report_studies: dict[str, str] = {}
        document_reports: dict[str, tuple[str, str]] = {}
        for basis in self.study_design_bases:
            for evidence in basis.evidence:
                if report_studies.setdefault(evidence.report_id, basis.study_id) != basis.study_id:
                    raise ValueError("benchmark design Report cannot bind multiple Studies")
                report_contract = (evidence.report_id, evidence.document_role)
                if (
                    document_reports.setdefault(evidence.document_id, report_contract)
                    != report_contract
                ):
                    raise ValueError(
                        "benchmark design document cannot declare multiple Report identities"
                    )
        if score_eligible_without_design and set(design_ids) != set(self.contributing_study_ids):
            raise ValueError(
                "otherwise score-eligible case must close every contributing Study design"
            )
        score_eligible = score_eligible_without_design and bool(design_ids)
        if score_eligible and (self.risk_of_bias_profile is None or self.evidence_pico is None):
            raise ValueError(
                "score-eligible benchmark case requires RoB and evidence PICO profiles"
            )
        if not score_eligible and design_ids:
            raise ValueError("non-scored case cannot carry unused Study design bases")
        if score_eligible:
            if self.synthesis.effect_measure not in {"RR", "OR"}:
                raise ValueError("score-eligible EC V1 case requires RR or OR")
            if self.certainty_target.threshold_value != 1.0:
                raise ValueError("score-eligible EC V1 case requires null threshold 1")
            study_effect_fact = next(
                (fact for fact in self.evidence_facts if fact.fact_id == "study-effects"),
                None,
            )
            prefix = "study_effects="
            if study_effect_fact is None or not study_effect_fact.quote.startswith(prefix):
                raise ValueError("score-eligible case requires structured study effects")
            try:
                study_effects = json.loads(study_effect_fact.quote.removeprefix(prefix))
                study_rows = study_effects["studies"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError("benchmark study effects are not valid JSON") from error
            if study_effects.get("effect_measure") != self.synthesis.effect_measure:
                raise ValueError("study-effect measure differs from published synthesis")
            if set(study_rows) != set(self.contributing_study_ids):
                raise ValueError("study effects must close every contributing Study")
            for row in study_rows.values():
                if set(row) != {"estimate", "ci_lower", "ci_upper"}:
                    raise ValueError("study-effect row must contain estimate and CI")
                if not row["ci_lower"] <= row["estimate"] <= row["ci_upper"]:
                    raise ValueError("study-effect estimate must lie inside its CI")
        if not score_eligible and (
            self.risk_of_bias_profile is not None or self.evidence_pico is not None
        ):
            raise ValueError("source-unclosed benchmark case cannot carry unused rating profiles")
        if self.risk_of_bias_profile is not None:
            if self.risk_of_bias_profile.study_judgments != self.study_risk_of_bias:
                raise ValueError("benchmark RoB profile differs from contributing Study map")
            if (
                self.risk_of_bias_profile.contribution_weight_percent
                != self.contribution_weight_percent
            ):
                raise ValueError("benchmark RoB profile differs from contribution weights")
        has_scope_source = (
            self.source_closure_document_id is not None and self.source_closure_quote is not None
        )
        if self.source_closure == "CLOSED" and (
            self.source_closure_document_id is not None or self.source_closure_quote is not None
        ):
            raise ValueError("closed benchmark source does not accept scope evidence")
        if self.source_closure != "CLOSED" and not has_scope_source:
            raise ValueError("non-closed benchmark source requires exact scope evidence")
        if (
            self.source_closure != "CLOSED"
            and self.source_closure_document_id not in self.source_derivations
        ):
            raise ValueError("non-closed benchmark source requires upstream scope lineage")
        if self.source_closure != "CLOSED":
            prefix = "scope_evidence="
            if self.source_closure_quote is None or not self.source_closure_quote.startswith(
                prefix
            ):
                raise ValueError("benchmark scope evidence must use the canonical prefix")
            try:
                scope_payload = json.loads(self.source_closure_quote.removeprefix(prefix))
                observations = scope_payload["observations"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError("benchmark scope evidence is not valid JSON") from error
            if scope_payload.get("case_id") != self.case_id or not observations:
                raise ValueError("benchmark scope evidence identity or observations are invalid")
            required_scope = {f"scope.observations[{index}]" for index in range(len(observations))}
            if output_fields.get(self.source_closure_document_id) != required_scope:
                raise ValueError("benchmark scope lineage must cover every observation exactly")
        scope_fields = [
            field for evidence in self.scope_basis_evidence for field in evidence.fields
        ]
        if len(scope_fields) != len(set(scope_fields)) or set(scope_fields) != {
            "selected_outcome_ids",
            "review_role",
            "comparison_kind",
            "synthesis_kind",
            "outcome_kind",
        }:
            raise ValueError("benchmark scope evidence must bind every gate field exactly once")
        if self.scope_basis.selected_outcome_ids != (self.selected_analysis_id,):
            raise ValueError("benchmark scope basis selected outcome differs")
        if any(
            representations[evidence.document_id] != "RAW_SOURCE"
            for evidence in self.scope_basis_evidence
        ):
            raise ValueError("benchmark scope basis must terminate at raw Source Documents")
        if (
            self.source_closure_document_id is not None
            and self.source_closure_document_id not in known_documents
        ):
            raise ValueError("scope evidence references an unknown Source Document")
        return self


class BenchmarkCatalog(EvidenceCertaintyModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    benchmark_id: Literal["metarigor-evidence-certainty-benchmark-v1"]
    grade_rule_version: Literal["CORE_GRADE_2025_V1"]
    data_extraction_inventory_sha256: Sha256
    cases: tuple[BenchmarkCaseDefinition, ...] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def ten_cases_are_unique(self) -> BenchmarkCatalog:
        case_ids = [item.case_id for item in self.cases]
        legacy_ids = [item.legacy_case_id for item in self.cases]
        if len(case_ids) != len(set(case_ids)) or len(legacy_ids) != len(set(legacy_ids)):
            raise ValueError("benchmark case ids must be unique")
        return self


class GoldDomain(EvidenceCertaintyModel):
    domain: GradeDomain
    judgment: DomainJudgment
    rationale: str = Field(min_length=1, max_length=8_000)
    evidence_fact_ids: tuple[str, ...] = Field(min_length=1, max_length=50)


class GoldCase(EvidenceCertaintyModel):
    case_id: str
    selected_analysis_id: str
    disposition: Literal["SCORED", "CORRECT_ABSTENTION_SOURCE_NOT_CLOSED"]
    expected_scope_disposition: Literal[
        "ELIGIBLE_SCORED_HEADLINE",
        "ELIGIBLE_SCORED_HEADLINE_WITH_SOURCE_CONFLICT",
        "SOURCE_NOT_CLOSED_REFERENCE_CONFLICT",
        "SOURCE_NOT_CLOSED_UNSUPPORTED_SYNTHESIS_METHOD",
        "SOURCE_NOT_CLOSED_SOURCE_GATED",
    ]
    domains: tuple[GoldDomain, ...] = Field(default=(), max_length=5)
    overall_downgrade_levels: int | None = Field(default=None, ge=0, le=3)
    final_certainty: Certainty | None = None
    overall_rationale: str = Field(min_length=1, max_length=8_000)
    curator_confidence: Literal["HIGH", "MODERATE", "LOW"]

    @model_validator(mode="after")
    def gold_respects_core_grade_bounds(self) -> GoldCase:
        if self.disposition == "CORRECT_ABSTENTION_SOURCE_NOT_CLOSED":
            if self.expected_scope_disposition in {
                "ELIGIBLE_SCORED_HEADLINE",
                "ELIGIBLE_SCORED_HEADLINE_WITH_SOURCE_CONFLICT",
            }:
                raise ValueError("abstention gold requires a source-closure disposition")
            if self.domains or self.overall_downgrade_levels is not None or self.final_certainty:
                raise ValueError("abstention gold cannot contain GRADE ratings")
            return self
        if self.expected_scope_disposition not in {
            "ELIGIBLE_SCORED_HEADLINE",
            "ELIGIBLE_SCORED_HEADLINE_WITH_SOURCE_CONFLICT",
        }:
            raise ValueError("scored gold requires an eligible scope disposition")
        if len(self.domains) != 5:
            raise ValueError("scored gold requires five domains")
        domains = [item.domain for item in self.domains]
        if len(domains) != len(set(domains)):
            raise ValueError("gold domains must be unique")
        lower = min(3, sum(judgment_bounds(item.judgment)[0] for item in self.domains))
        upper = min(3, sum(judgment_bounds(item.judgment)[1] for item in self.domains))
        if self.overall_downgrade_levels is None or not (
            lower <= self.overall_downgrade_levels <= upper
        ):
            raise ValueError("gold overall downgrade is outside domain close-call bounds")
        if self.final_certainty != final_certainty(self.overall_downgrade_levels):
            raise ValueError("gold final certainty does not match overall downgrade")
        return self


class BenchmarkGold(EvidenceCertaintyModel):
    schema_version: Literal["1.1.0"] = "1.1.0"
    benchmark_id: Literal["metarigor-evidence-certainty-benchmark-v1"]
    candidate_catalog_sha256: Sha256
    grade_rule_version: Literal["CORE_GRADE_2025_V1"]
    reference_maturity: Literal["CURATOR_ADJUDICATED_NOT_CLINICAL_EXPERT_VALIDATED"]
    external_comparator_blinded_during_adjudication: Literal[True]
    ois_alpha: Literal[0.05]
    ois_beta: Literal[0.2]
    ois_modest_relative_change: Literal[0.2]
    cases: tuple[GoldCase, ...] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def frozen_denominators_are_exact(self) -> BenchmarkGold:
        scored = tuple(item for item in self.cases if item.disposition == "SCORED")
        abstained = tuple(
            item
            for item in self.cases
            if item.disposition == "CORRECT_ABSTENTION_SOURCE_NOT_CLOSED"
        )
        if len(scored) != 7 or len(abstained) != 3:
            raise ValueError(
                "EC V1 gold must freeze exactly seven scored and three abstained cases"
            )
        if sum(len(item.domains) for item in scored) != 35:
            raise ValueError("EC V1 gold must freeze exactly 35 domain judgments")
        return self


