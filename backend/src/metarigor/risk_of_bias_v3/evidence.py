from __future__ import annotations

import re
from difflib import SequenceMatcher
from .method import METHOD_SPEC
from .models import AbsenceSearchHit, AbsenceSearchRecord, Domain, EvidenceQuoteCandidate, PreparedDocument, RoBSourceSpan, StudySourceBundle
QUESTION_PROBES = {
    "1.1": ("randomization_sequence",),
    "1.2": ("allocation_concealment",),
    "1.3": ("baseline_imbalance",),
    "2.1": ("participant_awareness",),
    "2.2": ("personnel_awareness",),
    "2.3": ("deviations",),
    "2.4": ("deviations",),
    "2.5": ("deviations",),
    "2.6": ("assignment_analysis",),
    "2.7": ("assignment_analysis",),
    "3.1": ("missing_outcome",),
    "3.2": ("missing_outcome",),
    "3.3": ("missing_outcome",),
    "3.4": ("missing_outcome",),
    "4.1": ("outcome_measurement",),
    "4.2": ("outcome_measurement",),
    "4.3": ("outcome_assessor_blinding",),
    "4.4": ("outcome_assessor_blinding",),
    "4.5": ("outcome_assessor_blinding",),
    "5.1": ("prespecified_analysis",),
    "5.2": ("multiple_measurements",),
    "5.3": ("multiple_analyses",),
}


DOMAIN_PROBES = {
    domain: tuple(
        dict.fromkeys(
            probe
            for question, probes in QUESTION_PROBES.items()
            if question.startswith(domain[1] + ".")
            for probe in probes
        )
    )
    for domain in ("D1", "D2", "D3", "D4", "D5")
}


def _nearest_text(text: str, quote: str, width: int = 240) -> str:
    words = quote.split()
    if not words:
        return ""
    probes = [" ".join(words[index : index + 6]) for index in range(max(1, len(words) - 5))]
    candidates: list[tuple[float, int]] = []
    lowered = text.lower()
    for probe in probes:
        token = probe.lower()
        for match in re.finditer(re.escape(token[: max(8, min(32, len(token)))]), lowered):
            start = max(0, match.start() - width // 2)
            excerpt = text[start : start + width]
            similarity = SequenceMatcher(None, quote.lower(), excerpt.lower()).ratio()
            candidates.append((similarity, start))
    if not candidates:
        return "no overlapping text found"
    _, start = max(candidates)
    return f"offset={start}: {text[start : start + width]}"


def _normalized_projection(value: str) -> str:
    return " ".join(value.replace("-\n", "").split()).casefold()


def resolve_quote(
    candidate: EvidenceQuoteCandidate,
    document: PreparedDocument,
) -> RoBSourceSpan:
    if candidate.document_key != document.document_key:
        raise ValueError("quote candidate uses another document")
    occurrences = [
        match.start() for match in re.finditer(re.escape(candidate.quote), document.full_text)
    ]
    if len(occurrences) == 1:
        start = occurrences[0]
        end = start + len(candidate.quote)
        return RoBSourceSpan(
            document_key=document.document_key,
            source_sha256=document.source_sha256,
            representation_sha256=document.representation_sha256,
            start_offset=start,
            end_offset=end,
            exact_text=document.full_text[start:end],
            match_kind="EXACT",
        )
    normalized_quote = _normalized_projection(candidate.quote)
    normalized_text = _normalized_projection(document.full_text)
    kind = (
        "NORMALIZED"
        if normalized_quote and normalized_text.count(normalized_quote) == 1
        else "UNRESOLVED"
    )
    return RoBSourceSpan(
        document_key=document.document_key,
        source_sha256=document.source_sha256,
        representation_sha256=document.representation_sha256,
        start_offset=0,
        end_offset=0,
        exact_text="",
        match_kind=kind,
        nearest_text=(
            "quote matched only after normalization"
            if kind == "NORMALIZED"
            else _nearest_text(document.full_text, candidate.quote)
        ),
    )


def build_absence_search_record(
    *,
    probe_id: str,
    study_key: str,
    domain: Domain,
    bundle: StudySourceBundle,
    required_roles_closed: bool,
) -> AbsenceSearchRecord:
    probes = METHOD_SPEC["absence_probes"]
    if probe_id not in probes or probe_id not in DOMAIN_PROBES[domain]:
        raise ValueError(f"unknown absence probe: {probe_id}")
    terms = tuple(probes[probe_id])
    hits: list[AbsenceSearchHit] = []
    for document in bundle.members:
        lowered = document.full_text.casefold()
        for term in terms:
            for match in re.finditer(re.escape(term.casefold()), lowered):
                start, end = match.span()
                hits.append(
                    AbsenceSearchHit(
                        document_key=document.document_key,
                        term=term,
                        start_offset=start,
                        end_offset=end,
                        exact_text=document.full_text[start:end],
                    )
                )
    if hits:
        conclusion = "HIT_REQUIRES_REVIEW"
    elif required_roles_closed:
        conclusion = "NO_HIT"
    else:
        conclusion = "SEARCH_INCOMPLETE"
    return AbsenceSearchRecord(
        probe_id=probe_id,
        study_key=study_key,
        domain=domain,
        terms=terms,
        searched_document_keys=tuple(item.document_key for item in bundle.members),
        searched_sections=("FULL_DOCUMENT",),
        hit_count=len(hits),
        hits=tuple(hits),
        bundle_sha256=bundle.bundle_sha256,
        conclusion=conclusion,
    )


