from __future__ import annotations

from collections.abc import Mapping
from .models import BenchmarkV2EvidenceMetrics, BenchmarkV2PredictionCase, BenchmarkV2PredictionDomain, ReviewGoldCase
def evaluate_answer_evidence(
    *,
    predictions: tuple[BenchmarkV2PredictionCase, ...],
    references: tuple[ReviewGoldCase, ...],
    partial_domains: Mapping[str, tuple[BenchmarkV2PredictionDomain, ...]] | None = None,
) -> BenchmarkV2EvidenceMetrics:
    """Uniformly evaluate answer-level evidence-claim closure for MR direct/staged outputs."""

    prediction_by_id = {item.case_id: item for item in predictions}
    partial_domains = partial_domains or {}
    reference_by_id = {item.case_id: item for item in references}
    if len(reference_by_id) != 20:
        raise ValueError("EC V2 evidence evaluator requires the fixed 20-case reference")
    reference_link_count = sum(
        len(domain.evidence_anchor_ids) for case in references for domain in case.domains
    )
    if reference_link_count != 141:
        raise ValueError("EC V2 evidence evaluator requires the frozen 141 reference links")

    assessed_domains = 0
    covered_domains = 0
    selected_claims = 0
    closed_claims = 0
    raw_closed_claims = 0
    unsupported = 0
    primary_closed = 0
    review_only = 0
    complete_cases = 0
    recalled_reference_links = 0

    for case_id, reference in reference_by_id.items():
        prediction = prediction_by_id.get(case_id)
        domains = prediction.domains if prediction is not None else partial_domains.get(case_id, ())
        if not domains:
            continue
        predicted_domains = {item.domain: item for item in domains}
        case_domains_complete = True
        for domain in domains:
            assessed_domains += 1
            selected_claims += len(domain.evidence_anchor_ids)
            supported = [
                item
                for item in domain.evidence_claims
                if item.support_status == "SUPPORTED" and item.source_span is not None
            ]
            closed_claims += len(supported)
            unsupported += sum(
                item.support_status != "SUPPORTED" for item in domain.evidence_claims
            )
            unsupported += max(0, len(domain.evidence_anchor_ids) - len(domain.evidence_claims))
            if supported:
                covered_domains += 1
            else:
                case_domains_complete = False
            for claim in supported:
                if claim.source_lineage == "PRIMARY_SOURCE_CLOSED":
                    raw_closed_claims += 1
                if claim.source_lineage == "PRIMARY_SOURCE_CLOSED":
                    primary_closed += 1
                if claim.source_lineage == "REVIEW_DOCUMENT_ONLY":
                    review_only += 1

        for reference_domain in reference.domains:
            predicted = predicted_domains.get(reference_domain.domain)
            if predicted is None:
                continue
            supported_ids = {
                item.evidence_id
                for item in predicted.evidence_claims
                if item.support_status == "SUPPORTED"
            }
            recalled_reference_links += sum(
                evidence_id in supported_ids for evidence_id in reference_domain.evidence_anchor_ids
            )

        if prediction is None:
            continue
        domain_union = tuple(
            dict.fromkeys(
                claim.evidence_id
                for domain in prediction.domains
                for claim in domain.evidence_claims
                if claim.support_status == "SUPPORTED" and claim.evidence_id is not None
            )
        )
        expected_spans = {
            claim.source_span.span_id
            for domain in prediction.domains
            for claim in domain.evidence_claims
            if claim.support_status == "SUPPORTED" and claim.source_span is not None
        }
        actual_spans = {item.span_id for item in prediction.overall_source_spans}
        overall_complete = (
            prediction.overall_evidence_anchor_ids == domain_union
            and actual_spans == expected_spans
            and prediction.evidence_verification is not None
            and prediction.evidence_verification.verdict == "PASSED"
        )
        complete_cases += int(case_domains_complete and overall_complete)

    return BenchmarkV2EvidenceMetrics(
        answer_evidence_coverage_numerator=covered_domains,
        answer_evidence_coverage_denominator=assessed_domains,
        source_span_closure_numerator=closed_claims,
        source_span_closure_denominator=selected_claims,
        reference_anchor_recall_numerator=recalled_reference_links,
        raw_source_closure_numerator=raw_closed_claims,
        raw_source_closure_denominator=closed_claims,
        unsupported_evidence_count=unsupported,
        case_evidence_complete_numerator=complete_cases,
        primary_source_closed_selected_span_count=primary_closed,
        review_document_only_selected_span_count=review_only,
    )

