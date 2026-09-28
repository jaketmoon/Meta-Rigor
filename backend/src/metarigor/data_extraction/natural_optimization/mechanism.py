from __future__ import annotations

"""Wrap citations in existing full text and deterministically retrieve Source Spans without selecting or revising values."""


import hashlib
import re
from pydantic import BaseModel, ConfigDict, Field
from metarigor.data_extraction.models import SourceSpan
from metarigor.local_run import canonical_json
VERSION = "natural-source-citations-v1"


SELECTION_INSTRUCTION = """Distinguish source observations from the requested result:
For a reported result, retain the source's outcome label, timepoint, analysis population,
statistic type, and comparison-arm labels in your readable account. Relate that observation
to the outcome requested in the Protocol Source. Apply a selection rule only when supported
by that Protocol; do not invent a nearest-timepoint, population, or report-priority rule.
When several source observations compete for the same requested result, show their identities
and numbers separately. State which observation the Protocol selects, or explicitly preserve
the unresolved ambiguity if it does not select one. Keep the final summary consistent with
these decisions; do not mark an unresolved or unreported result as uniquely reported there.
A reported group size belongs to its source analysis population and outcome. Do not silently
substitute an overall randomized N for an outcome-specific denominator or infer a denominator
from a different outcome. Preserve partial values when the complete result is unavailable.
Preserve the source statistic: median/IQR, mean/SD, event counts and continuous durations are
different observations. Do not relabel a related outcome or statistic as the requested one.
These are presentation and selection requirements within the same extraction call, not a
request for a new screening stage, a critique, confidence scores, or statistical synthesis.
"""


class CitationUnit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    citation_id: str = Field(pattern=r"^S[0-9]{6}$")
    unit_id: str
    document_key: str
    report_key: str
    view_path: str
    view_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    quote: str = Field(min_length=1)
    quote_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bind_citations(candidate: str, citation_map: list[dict]) -> dict:
    """Close only selected citations; marker positions are not semantic anchors for complete results or fields."""
    entries = [CitationUnit.model_validate(item) for item in citation_map]
    by_id = {item.citation_id: item for item in entries}
    if len(by_id) != len(entries):
        raise ValueError("duplicate citation map ID")
    for item in entries:
        if (
            item.end_char - item.start_char != len(item.quote)
            or _sha(item.quote) != item.quote_sha256
        ):
            raise ValueError("citation map quote integrity failure")
    spans, issues = [], []
    markers = list(re.finditer(r"\[src:([^\]\r\n]*)\]", candidate))
    if not markers:
        issues.append({"code": "NO_SOURCE_CITATIONS"})
    for marker in markers:
        ids = [part.strip() for part in marker.group(1).split(",")]
        for identifier in ids:
            context = {
                "citation_id": identifier,
                "candidate_start_char": marker.start(),
                "candidate_end_char": marker.end(),
                "candidate_quote": marker.group(0),
            }
            entry = by_id.get(identifier)
            if entry is None:
                issues.append({**context, "code": "UNKNOWN_SOURCE_CITATION"})
                continue
            if len(entry.quote) > 20_000:
                issues.append({**context, "code": "SOURCE_UNIT_EXCEEDS_SPAN_BOUND"})
                continue
            span = SourceSpan(
                span_id=f"span:{entry.unit_id}",
                document_key=entry.document_key,
                view_path=entry.view_path,
                view_sha256=entry.view_sha256,
                start_char=entry.start_char,
                end_char=entry.end_char,
                quote=entry.quote,
            )
            spans.append(
                {
                    **context,
                    "source_span": span.model_dump(mode="json"),
                    "report_key": entry.report_key,
                    "quote_sha256": entry.quote_sha256,
                }
            )
    # Malformed markers are observable failures too; do not seek replacement evidence elsewhere.
    covered = {marker.start() for marker in markers}
    for malformed in re.finditer(r"\[src:", candidate):
        if malformed.start() not in covered:
            issues.append(
                {"code": "MALFORMED_SOURCE_CITATION", "candidate_start_char": malformed.start()}
            )
    return {
        "schema_version": VERSION,
        "candidate_sha256": _sha(candidate),
        "citation_map_sha256": hashlib.sha256(canonical_json(citation_map)).hexdigest(),
        "spans": spans,
        "issues": issues,
        "semantic_field_binding_evaluated": False,
    }

