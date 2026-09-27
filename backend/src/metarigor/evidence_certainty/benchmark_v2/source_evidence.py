from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from metarigor.evidence_certainty.benchmark_v1 import load_benchmark_catalog as load_v1_catalog
from metarigor.evidence_certainty.benchmark_v1.models import BenchmarkCaseDefinition
from metarigor.evidence_certainty.models import GRADE_DOMAINS, DomainJudgment, EvidenceFact, EvidenceFactKind, GradeDomain, SourceDocumentReference, SourceSpan
from metarigor.local_run import RunFolder
from .models import BenchmarkV2Catalog, BenchmarkV2EvidenceAnchorSidecar, BenchmarkV2EvidenceClaim, BenchmarkV2EvidenceVerification, BenchmarkV2EvidenceVerificationIssue, BenchmarkV2PredictionCase, BenchmarkV2PredictionDomain, ReviewBenchmarkCase, ReviewCaseSufficiency, ReviewFigureEvidence, ReviewTextEvidence
_REPO_ROOT = Path(__file__).resolve().parents[5]


_V2_DATASET_PREFIX = "backend/src/metarigor/evidence_certainty/benchmark_v2/dataset/"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def snapshot_path_map(*snapshots: Mapping[str, Any] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for snapshot in snapshots:
        if snapshot is None:
            continue
        for item in snapshot.get("source_artifacts", ()):
            source_path = item.get("source_path")
            run_path = item.get("path")
            if isinstance(source_path, str) and isinstance(run_path, str):
                result[source_path] = run_path
    return result


def _line_offset(content: str, line_number: int, quote: str) -> int:
    lines = content.splitlines(keepends=True)
    if not 1 <= line_number <= len(lines):
        raise ValueError("candidate evidence line is outside its Document View")
    raw_line = lines[line_number - 1]
    line = raw_line.rstrip("\r\n")
    if line != quote:
        raise ValueError("candidate evidence line differs from its frozen quote")
    return sum(len(item) for item in lines[: line_number - 1])


def _mapped_path(source_path: str, paths: Mapping[str, str]) -> str:
    try:
        return paths[source_path]
    except KeyError as error:
        raise ValueError(
            f"source artifact was not sealed into the Run Folder: {source_path}"
        ) from error


def _v1_reference(
    case: BenchmarkCaseDefinition,
    source,
    paths: Mapping[str, str],
) -> SourceDocumentReference:
    document = next(
        item for item in case.source_documents if item.document_id == source.document_id
    )
    return SourceDocumentReference(
        document_id=source.document_id,
        input_path=_mapped_path(document.path, paths),
        document_sha256=source.document_sha256,
        locator_kind=source.locator_kind,
        locator=source.locator,
        derivation=source.derivation,
        output_fields=source.output_fields,
    )


def _v1_sidecar(
    *,
    case: BenchmarkCaseDefinition,
    evidence_id: str,
    domain_relevance: tuple[GradeDomain, ...],
    statement: str,
    fact_kind: EvidenceFactKind,
    source_lineage: str,
    document_id: str,
    quote: str,
    paths: Mapping[str, str],
) -> BenchmarkV2EvidenceAnchorSidecar:
    document = next(item for item in case.source_documents if item.document_id == document_id)
    original = (_REPO_ROOT / document.path).read_bytes()
    if _sha256(original) != document.sha256:
        raise ValueError(f"V1 Source Document hash changed: {case.case_id}/{document_id}")
    text = original.decode("utf-8")
    if text.count(quote) != 1:
        raise ValueError(f"V1 evidence quote is not unique: {case.case_id}/{evidence_id}")
    start = text.index(quote)
    provenance = tuple(
        _v1_reference(case, item, paths) for item in case.source_derivations.get(document_id, ())
    )
    view_path = _mapped_path(document.path, paths)
    source_span = SourceSpan(
        span_id=f"span:{case.case_id}:{evidence_id}",
        document_id=document.document_id,
        input_path=view_path,
        document_sha256=document.sha256,
        start_offset=start,
        end_offset=start + len(quote),
        quote=quote,
        quote_sha256=_sha256(quote.encode("utf-8")),
        source_representation=document.source_representation,
        derivation_sources=provenance,
    )
    terminal_source = provenance[0] if provenance else None
    return BenchmarkV2EvidenceAnchorSidecar(
        evidence_id=evidence_id,
        domain_relevance=domain_relevance,
        statement=statement,
        evidence_fact=EvidenceFact(
            fact_id=evidence_id,
            fact_kind=fact_kind,
            statement=statement,
            domain_relevance=domain_relevance,
            source_span=source_span,
        ),
        source_lineage=source_lineage,
        source_path=terminal_source.input_path if terminal_source is not None else view_path,
        source_sha256=(
            terminal_source.document_sha256 if terminal_source is not None else document.sha256
        ),
        view_path=view_path,
        view_sha256=document.sha256,
        source_span=source_span,
        raw_source_provenance=provenance,
    )


def _lineage_gap_sidecar(
    *,
    evidence_id: str,
    domain_relevance: tuple[GradeDomain, ...],
    statement: str,
    error: Exception,
) -> BenchmarkV2EvidenceAnchorSidecar:
    return BenchmarkV2EvidenceAnchorSidecar(
        evidence_id=evidence_id,
        domain_relevance=domain_relevance,
        statement=statement,
        source_lineage="SOURCE_LINEAGE_GAP",
        issue_codes=(f"SOURCE_LINEAGE_GAP:{type(error).__name__}",),
        issue_detail=str(error)[:4_000],
    )


def _v1_sidecars(
    case: BenchmarkCaseDefinition,
    candidate_input: Any,
    paths: Mapping[str, str],
) -> tuple[BenchmarkV2EvidenceAnchorSidecar, ...]:
    definitions: dict[
        str, tuple[tuple[GradeDomain, ...], str, EvidenceFactKind, str, str, str]
    ] = {}
    for fact in case.evidence_facts:
        definitions[fact.fact_id] = (
            fact.domain_relevance,
            fact.statement,
            fact.fact_kind,
            "REVIEW_DOCUMENT_ONLY",
            fact.document_id,
            fact.quote,
        )
    for index, item in enumerate(case.scope_basis_evidence):
        definitions[f"scope-{index}"] = (
            GRADE_DOMAINS,
            item.quote,
            "PROTOCOL_TARGET",
            "REVIEW_DOCUMENT_ONLY",
            item.document_id,
            item.quote,
        )
    for basis in case.study_design_bases:
        for index, item in enumerate(basis.evidence):
            definitions[f"design-{basis.study_id}-{index}"] = (
                ("RISK_OF_BIAS", "INDIRECTNESS"),
                item.quote,
                "STUDY_DESIGN",
                "PRIMARY_SOURCE_CLOSED",
                item.document_id,
                item.quote,
            )
    expected_ids = tuple(item.evidence_id for item in candidate_input.evidence)
    if set(expected_ids) != set(definitions):
        raise ValueError(
            f"V1 candidate evidence differs from SourceSpan definitions: {case.case_id}"
        )
    sidecars = []
    for evidence_id in expected_ids:
        domains, statement, fact_kind, source_lineage, document_id, quote = definitions[evidence_id]
        try:
            sidecars.append(
                _v1_sidecar(
                    case=case,
                    evidence_id=evidence_id,
                    domain_relevance=domains,
                    statement=statement,
                    fact_kind=fact_kind,
                    source_lineage=source_lineage,
                    document_id=document_id,
                    quote=quote,
                    paths=paths,
                )
            )
        except (KeyError, OSError, ValueError) as error:
            sidecars.append(
                _lineage_gap_sidecar(
                    evidence_id=evidence_id,
                    domain_relevance=domains,
                    statement=statement,
                    error=error,
                )
            )
    return tuple(sidecars)


def _review_raw_reference(
    *,
    case: ReviewBenchmarkCase,
    raw_path: str,
    raw_sha256: str,
    paths: Mapping[str, str],
    locator: str,
    output_field: str,
) -> SourceDocumentReference:
    return SourceDocumentReference(
        document_id=f"{case.source.pmcid}-review-source",
        input_path=_mapped_path(raw_path, paths),
        document_sha256=raw_sha256,
        locator_kind="TEXT_SECTION",
        locator=locator,
        derivation="DIRECT_TRANSCRIPTION",
        output_fields=(output_field,),
    )


def _review_text_sidecar(
    *,
    case: ReviewBenchmarkCase,
    evidence_id: str,
    domain_relevance: tuple[GradeDomain, ...],
    statement: str,
    quote: str,
    line_number: int,
    candidate_content: str,
    candidate_path: str,
    candidate_sha256: str,
    raw_path: str,
    raw_sha256: str,
    paths: Mapping[str, str],
) -> BenchmarkV2EvidenceAnchorSidecar:
    start = _line_offset(candidate_content, line_number, quote)
    raw = _review_raw_reference(
        case=case,
        raw_path=raw_path,
        raw_sha256=raw_sha256,
        paths=paths,
        locator=(
            "raw Review JATS document; candidate-view derivation lineage only; "
            "exact raw locator not closed"
        ),
        output_field=f"evidence.{evidence_id}",
    )
    view_path = _mapped_path(candidate_path, paths)
    source_span = SourceSpan(
        span_id=f"span:{case.case_id}:{evidence_id}",
        document_id=f"{case.source.pmcid}-candidate-view",
        input_path=view_path,
        document_sha256=candidate_sha256,
        start_offset=start,
        end_offset=start + len(quote),
        quote=quote,
        quote_sha256=_sha256(quote.encode("utf-8")),
        source_representation="CURATED_DERIVATION",
        derivation_sources=(raw,),
    )
    return BenchmarkV2EvidenceAnchorSidecar(
        evidence_id=evidence_id,
        domain_relevance=domain_relevance,
        statement=statement,
        evidence_fact=EvidenceFact(
            fact_id=evidence_id,
            fact_kind=domain_relevance[0],
            statement=statement,
            domain_relevance=domain_relevance,
            source_span=source_span,
        ),
        source_lineage="REVIEW_DOCUMENT_ONLY",
        source_path=raw.input_path,
        source_sha256=raw.document_sha256,
        view_path=view_path,
        view_sha256=candidate_sha256,
        source_span=source_span,
        raw_source_provenance=(raw,),
    )


def _figure_references(
    *,
    case: ReviewBenchmarkCase,
    evidence: ReviewFigureEvidence,
    paths: Mapping[str, str],
) -> tuple[SourceDocumentReference, ...]:
    references = [
        SourceDocumentReference(
            document_id=f"{case.source.pmcid}-figure-{_sha256(evidence.artifact_path.encode())[:12]}",
            input_path=_mapped_path(evidence.artifact_path, paths),
            document_sha256=evidence.artifact_sha256,
            locator_kind="FIGURE_REGION",
            locator=f"{evidence.figure_label}; {evidence.source_url}",
            derivation="FIGURE_TRANSCRIPTION",
            output_fields=(f"evidence.{evidence.evidence_id}",),
        )
    ]
    if evidence.source_document_path is not None and evidence.source_document_sha256 is not None:
        references.append(
            SourceDocumentReference(
                document_id=(
                    f"{case.source.pmcid}-figure-container-"
                    f"{_sha256(evidence.source_document_path.encode())[:12]}"
                ),
                input_path=_mapped_path(evidence.source_document_path, paths),
                document_sha256=evidence.source_document_sha256,
                locator_kind="TEXT_SECTION",
                locator=evidence.derivation_locator or evidence.figure_label,
                derivation="DIRECT_TRANSCRIPTION",
                output_fields=(f"evidence.{evidence.evidence_id}",),
            )
        )
    return tuple(references)


def _review_figure_sidecar(
    *,
    folder: RunFolder,
    case: ReviewBenchmarkCase,
    domain: GradeDomain,
    evidence: ReviewFigureEvidence,
    paths: Mapping[str, str],
) -> BenchmarkV2EvidenceAnchorSidecar:
    view = folder.write_immutable(
        f"inputs/evidence-views/{case.case_id}/{evidence.evidence_id}.txt",
        evidence.transcription.encode("utf-8"),
    )
    references = _figure_references(case=case, evidence=evidence, paths=paths)
    source_span = SourceSpan(
        span_id=f"span:{case.case_id}:{evidence.evidence_id}",
        document_id=f"{case.source.pmcid}-{evidence.evidence_id}-view",
        input_path=view.path,
        document_sha256=view.sha256,
        start_offset=0,
        end_offset=len(evidence.transcription),
        quote=evidence.transcription,
        quote_sha256=_sha256(evidence.transcription.encode("utf-8")),
        source_representation="CURATED_DERIVATION",
        derivation_sources=references,
    )
    return BenchmarkV2EvidenceAnchorSidecar(
        evidence_id=evidence.evidence_id,
        domain_relevance=(domain,),
        statement=evidence.transcription,
        evidence_fact=EvidenceFact(
            fact_id=evidence.evidence_id,
            fact_kind=domain,
            statement=evidence.transcription,
            domain_relevance=(domain,),
            source_span=source_span,
        ),
        source_lineage="REVIEW_DOCUMENT_ONLY",
        source_path=references[0].input_path,
        source_sha256=references[0].document_sha256,
        view_path=view.path,
        view_sha256=view.sha256,
        source_span=source_span,
        raw_source_provenance=references,
    )


def _review_sidecars(
    *,
    folder: RunFolder,
    case: ReviewBenchmarkCase,
    candidate_input: Any,
    sufficiency: ReviewCaseSufficiency | None,
    paths: Mapping[str, str],
) -> tuple[BenchmarkV2EvidenceAnchorSidecar, ...]:
    manifest = json.loads((_REPO_ROOT / case.source.manifest_path).read_text(encoding="utf-8"))
    candidate_binding = manifest["candidate"]
    raw_binding = manifest["raw_source"]
    candidate_path = f"{_V2_DATASET_PREFIX}{candidate_binding['path']}"
    raw_path = f"{_V2_DATASET_PREFIX}{raw_binding['path']}"
    candidate_content = candidate_input.candidate_document
    if _sha256(candidate_content.encode("utf-8")) != candidate_binding["sha256"]:
        raise ValueError(f"Review candidate Document View changed: {case.case_id}")
    input_by_id = {item.evidence_id: item for item in candidate_input.evidence}
    sidecars: list[BenchmarkV2EvidenceAnchorSidecar] = []
    for anchor in case.anchors:
        item = input_by_id[anchor.anchor_id]
        try:
            sidecars.append(
                _review_text_sidecar(
                    case=case,
                    evidence_id=anchor.anchor_id,
                    domain_relevance=anchor.domain_relevance,
                    statement=item.quote,
                    quote=item.quote,
                    line_number=anchor.line_number,
                    candidate_content=candidate_content,
                    candidate_path=candidate_path,
                    candidate_sha256=candidate_binding["sha256"],
                    raw_path=raw_path,
                    raw_sha256=raw_binding["sha256"],
                    paths=paths,
                )
            )
        except (KeyError, OSError, ValueError) as error:
            sidecars.append(
                _lineage_gap_sidecar(
                    evidence_id=anchor.anchor_id,
                    domain_relevance=anchor.domain_relevance,
                    statement=item.quote,
                    error=error,
                )
            )
    for domain in sufficiency.domains if sufficiency is not None else ():
        for evidence in domain.supplemental_evidence:
            if isinstance(evidence, ReviewTextEvidence):
                try:
                    sidecars.append(
                        _review_text_sidecar(
                            case=case,
                            evidence_id=evidence.evidence_id,
                            domain_relevance=(domain.domain,),
                            statement=evidence.quote,
                            quote=evidence.quote,
                            line_number=evidence.line_number,
                            candidate_content=candidate_content,
                            candidate_path=candidate_path,
                            candidate_sha256=candidate_binding["sha256"],
                            raw_path=raw_path,
                            raw_sha256=raw_binding["sha256"],
                            paths=paths,
                        )
                    )
                except (KeyError, OSError, ValueError) as error:
                    sidecars.append(
                        _lineage_gap_sidecar(
                            evidence_id=evidence.evidence_id,
                            domain_relevance=(domain.domain,),
                            statement=evidence.quote,
                            error=error,
                        )
                    )
                continue
            if not isinstance(evidence, ReviewFigureEvidence):
                raise TypeError("unsupported review evidence representation")
            try:
                sidecars.append(
                    _review_figure_sidecar(
                        folder=folder,
                        case=case,
                        domain=domain.domain,
                        evidence=evidence,
                        paths=paths,
                    )
                )
            except (KeyError, OSError, ValueError) as error:
                sidecars.append(
                    _lineage_gap_sidecar(
                        evidence_id=evidence.evidence_id,
                        domain_relevance=(domain.domain,),
                        statement=evidence.transcription,
                        error=error,
                    )
                )
    identities = [item.evidence_id for item in sidecars]
    if len(identities) != len(set(identities)):
        raise ValueError(f"Review evidence ids are not unique across domains: {case.case_id}")
    return tuple(sidecars)


def materialize_candidate_sidecars(
    *,
    folder: RunFolder,
    catalog: BenchmarkV2Catalog,
    candidates: Sequence[Any],
    review_contract: Any | None,
    source_paths: Mapping[str, str],
) -> dict[str, tuple[BenchmarkV2EvidenceAnchorSidecar, ...]]:
    v1_cases = {item.case_id: item for item in load_v1_catalog().cases}
    review_cases = {item.case_id: item for item in catalog.review_cases}
    sufficiency = (
        {item.case_id: item for item in review_contract.cases}
        if review_contract is not None
        else {}
    )
    result: dict[str, tuple[BenchmarkV2EvidenceAnchorSidecar, ...]] = {}
    for prepared in candidates:
        if prepared.case_id in v1_cases:
            sidecars = _v1_sidecars(v1_cases[prepared.case_id], prepared.input, source_paths)
        else:
            sidecars = _review_sidecars(
                folder=folder,
                case=review_cases[prepared.case_id],
                candidate_input=prepared.input,
                sufficiency=sufficiency.get(prepared.case_id),
                paths=source_paths,
            )
        expected_ids = {item.evidence_id for item in prepared.input.evidence}
        if review_contract is not None and prepared.case_id in sufficiency:
            expected_ids.update(
                item.evidence_id
                for domain in sufficiency[prepared.case_id].domains
                for item in domain.supplemental_evidence
            )
        if {item.evidence_id for item in sidecars} != expected_ids:
            raise ValueError(
                f"SourceSpan sidecar does not close candidate evidence: {prepared.case_id}"
            )
        artifact = folder.write_immutable(
            f"inputs/evidence-sidecars/{prepared.case_id}.json",
            json.dumps(
                [item.model_dump(mode="json") for item in sidecars],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        result[prepared.case_id] = sidecars
        folder.write_immutable(
            f"inputs/evidence-sidecars/{prepared.case_id}.seal.json",
            json.dumps(
                {"path": artifact.path, "sha256": artifact.sha256, "count": len(sidecars)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
    return result


def _issue(
    code: str,
    *,
    case_id: str,
    domain: GradeDomain | None,
    evidence_id: str | None,
    detail: str,
) -> BenchmarkV2EvidenceVerificationIssue:
    return BenchmarkV2EvidenceVerificationIssue(
        issue_code=code,
        case_id=case_id,
        domain=domain,
        evidence_id=evidence_id,
        detail=detail,
    )


def _verify_sidecar(
    sidecar: BenchmarkV2EvidenceAnchorSidecar,
    *,
    folder: RunFolder,
    case_id: str,
    domain: GradeDomain,
) -> tuple[BenchmarkV2EvidenceVerificationIssue, ...]:
    if sidecar.source_lineage == "SOURCE_LINEAGE_GAP":
        return (
            _issue(
                "SOURCE_LINEAGE_GAP",
                case_id=case_id,
                domain=domain,
                evidence_id=sidecar.evidence_id,
                detail=sidecar.issue_detail
                or "Candidate evidence has no deterministically closed SourceSpan sidecar.",
            ),
        )
    if (
        sidecar.view_path is None
        or sidecar.view_sha256 is None
        or sidecar.source_path is None
        or sidecar.source_sha256 is None
        or sidecar.source_span is None
    ):
        raise ValueError("closed sidecar is missing verifier inputs")
    issues: list[BenchmarkV2EvidenceVerificationIssue] = []
    try:
        view = folder.resolve(sidecar.view_path).read_bytes()
    except OSError as error:
        return (
            _issue(
                "VIEW_FILE_MISSING",
                case_id=case_id,
                domain=domain,
                evidence_id=sidecar.evidence_id,
                detail=str(error),
            ),
        )
    if _sha256(view) != sidecar.view_sha256:
        issues.append(
            _issue(
                "VIEW_SHA256_MISMATCH",
                case_id=case_id,
                domain=domain,
                evidence_id=sidecar.evidence_id,
                detail="Document View bytes differ from the sealed SHA-256.",
            )
        )
    try:
        text = view.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    span = sidecar.source_span
    if span.end_offset > len(text) or text[span.start_offset : span.end_offset] != span.quote:
        issues.append(
            _issue(
                "SOURCE_SPAN_OFFSET_MISMATCH",
                case_id=case_id,
                domain=domain,
                evidence_id=sidecar.evidence_id,
                detail="Document View start/end offsets do not resolve to the verbatim quote.",
            )
        )
    if _sha256(span.quote.encode("utf-8")) != span.quote_sha256:
        issues.append(
            _issue(
                "QUOTE_SHA256_MISMATCH",
                case_id=case_id,
                domain=domain,
                evidence_id=sidecar.evidence_id,
                detail="Verbatim quote differs from quote_sha256.",
            )
        )
    try:
        source = folder.resolve(sidecar.source_path).read_bytes()
    except OSError as error:
        issues.append(
            _issue(
                "SOURCE_FILE_MISSING",
                case_id=case_id,
                domain=domain,
                evidence_id=sidecar.evidence_id,
                detail=str(error),
            )
        )
    else:
        if _sha256(source) != sidecar.source_sha256:
            issues.append(
                _issue(
                    "SOURCE_SHA256_MISMATCH",
                    case_id=case_id,
                    domain=domain,
                    evidence_id=sidecar.evidence_id,
                    detail="Source Document bytes differ from the sealed SHA-256.",
                )
            )
    terminal_sources = {
        (item.input_path, item.document_sha256) for item in sidecar.raw_source_provenance
    }
    if span.source_representation == "RAW_SOURCE":
        terminal_sources.add((sidecar.view_path, sidecar.view_sha256))
    if (sidecar.source_path, sidecar.source_sha256) not in terminal_sources:
        issues.append(
            _issue(
                "SOURCE_IDENTITY_MISMATCH",
                case_id=case_id,
                domain=domain,
                evidence_id=sidecar.evidence_id,
                detail="Source Document identity is not a terminal raw provenance binding.",
            )
        )
    for reference in sidecar.raw_source_provenance:
        try:
            raw = folder.resolve(reference.input_path).read_bytes()
        except OSError as error:
            issues.append(
                _issue(
                    "SOURCE_FILE_MISSING",
                    case_id=case_id,
                    domain=domain,
                    evidence_id=sidecar.evidence_id,
                    detail=str(error),
                )
            )
            continue
        if _sha256(raw) != reference.document_sha256:
            issues.append(
                _issue(
                    "SOURCE_SHA256_MISMATCH",
                    case_id=case_id,
                    domain=domain,
                    evidence_id=sidecar.evidence_id,
                    detail=f"Raw provenance hash changed: {reference.document_id}",
                )
            )
    return tuple(issues)


def materialize_domain_prediction(
    *,
    folder: RunFolder,
    case_id: str,
    domain: GradeDomain,
    judgment: DomainJudgment,
    rationale: str,
    evidence_anchor_ids: Sequence[str],
    sidecars: Sequence[BenchmarkV2EvidenceAnchorSidecar],
) -> tuple[BenchmarkV2PredictionDomain, tuple[BenchmarkV2EvidenceVerificationIssue, ...]]:
    sidecar_by_id = {item.evidence_id: item for item in sidecars}
    claims: list[BenchmarkV2EvidenceClaim] = []
    issues: list[BenchmarkV2EvidenceVerificationIssue] = []
    seen: set[str] = set()
    for index, evidence_id in enumerate(evidence_anchor_ids):
        claim_issues: list[BenchmarkV2EvidenceVerificationIssue] = []
        sidecar = sidecar_by_id.get(evidence_id)
        if evidence_id in seen:
            claim_issues.append(
                _issue(
                    "DUPLICATE_EVIDENCE_ID",
                    case_id=case_id,
                    domain=domain,
                    evidence_id=evidence_id,
                    detail="The answer selected the same evidence ID more than once.",
                )
            )
        seen.add(evidence_id)
        if sidecar is None:
            claim_issues.append(
                _issue(
                    "UNKNOWN_EVIDENCE_ID",
                    case_id=case_id,
                    domain=domain,
                    evidence_id=evidence_id,
                    detail="The answer selected an evidence ID absent from the sealed sidecar.",
                )
            )
        elif domain not in sidecar.domain_relevance:
            claim_issues.append(
                _issue(
                    "EVIDENCE_DOMAIN_MISMATCH",
                    case_id=case_id,
                    domain=domain,
                    evidence_id=evidence_id,
                    detail="The selected evidence is not marked relevant to this domain.",
                )
            )
        elif not claim_issues:
            claim_issues.extend(
                _verify_sidecar(sidecar, folder=folder, case_id=case_id, domain=domain)
            )
        issues.extend(claim_issues)
        supported = sidecar is not None and not claim_issues
        claims.append(
            BenchmarkV2EvidenceClaim(
                claim_id=f"{case_id}:{domain}:{index}:{evidence_id}",
                case_id=case_id,
                domain=domain,
                claim_text=rationale,
                evidence_id=evidence_id,
                support_status="SUPPORTED" if supported else "UNSUPPORTED",
                source_lineage=(sidecar.source_lineage if supported else "SOURCE_LINEAGE_GAP"),
                source_path=sidecar.source_path if supported else None,
                source_sha256=sidecar.source_sha256 if supported else None,
                view_path=sidecar.view_path if supported else None,
                view_sha256=sidecar.view_sha256 if supported else None,
                source_span=sidecar.source_span if supported else None,
                issue_codes=tuple(item.issue_code for item in claim_issues),
            )
        )
    if not any(item.support_status == "SUPPORTED" for item in claims):
        issues.append(
            _issue(
                "DOMAIN_SOURCE_SPAN_MISSING",
                case_id=case_id,
                domain=domain,
                evidence_id=None,
                detail="An assessed domain has no selected evidence with a valid Source Span.",
            )
        )
    return (
        BenchmarkV2PredictionDomain(
            domain=domain,
            judgment=judgment,
            rationale=rationale,
            evidence_anchor_ids=tuple(evidence_anchor_ids),
            evidence_claims=tuple(claims),
        ),
        tuple(issues),
    )


def materialize_prediction_case(
    *,
    case_id: str,
    domains: Sequence[BenchmarkV2PredictionDomain],
    overall_downgrade_levels: int,
    final_certainty: str,
    overall_rationale: str,
    domain_issues: Sequence[BenchmarkV2EvidenceVerificationIssue] = (),
) -> BenchmarkV2PredictionCase:
    overall_ids = tuple(
        dict.fromkeys(
            claim.evidence_id
            for domain in domains
            for claim in domain.evidence_claims
            if claim.support_status == "SUPPORTED" and claim.evidence_id is not None
        )
    )
    spans_by_id = {
        claim.source_span.span_id: claim.source_span
        for domain in domains
        for claim in domain.evidence_claims
        if claim.support_status == "SUPPORTED" and claim.source_span is not None
    }
    overall_spans = tuple(spans_by_id.values())
    issues = list(domain_issues)
    issues.extend(
        verify_overall_evidence(
            case_id=case_id,
            domains=domains,
            overall_evidence_anchor_ids=overall_ids,
            overall_source_spans=overall_spans,
        )
    )
    verification = BenchmarkV2EvidenceVerification(
        case_id=case_id,
        verdict="PASSED" if not issues else "FAILED",
        issues=tuple(issues),
    )
    return BenchmarkV2PredictionCase(
        case_id=case_id,
        domains=tuple(domains),
        overall_downgrade_levels=overall_downgrade_levels,
        final_certainty=final_certainty,
        overall_rationale=overall_rationale,
        overall_evidence_anchor_ids=overall_ids,
        overall_source_spans=overall_spans,
        evidence_verification=verification,
    )


def verify_overall_evidence(
    *,
    case_id: str,
    domains: Sequence[BenchmarkV2PredictionDomain],
    overall_evidence_anchor_ids: Sequence[str],
    overall_source_spans: Sequence[SourceSpan],
) -> tuple[BenchmarkV2EvidenceVerificationIssue, ...]:
    """检查 overall 只能引用五个 domain 已闭合 evidence 的并集。"""

    expected_ids = tuple(
        dict.fromkeys(
            claim.evidence_id
            for domain in domains
            for claim in domain.evidence_claims
            if claim.support_status == "SUPPORTED" and claim.evidence_id is not None
        )
    )
    expected_spans = tuple(
        dict.fromkeys(
            claim.source_span
            for domain in domains
            for claim in domain.evidence_claims
            if claim.support_status == "SUPPORTED" and claim.source_span is not None
        )
    )
    issues: list[BenchmarkV2EvidenceVerificationIssue] = []
    if any(evidence_id not in expected_ids for evidence_id in overall_evidence_anchor_ids):
        issues.append(
            _issue(
                "OVERALL_EVIDENCE_OUTSIDE_DOMAIN_UNION",
                case_id=case_id,
                domain=None,
                evidence_id=None,
                detail="Overall evidence contains an ID outside closed domain evidence.",
            )
        )
    if tuple(overall_source_spans) != expected_spans:
        issues.append(
            _issue(
                "OVERALL_SOURCE_SPAN_MISMATCH",
                case_id=case_id,
                domain=None,
                evidence_id=None,
                detail="Overall SourceSpans differ from closed domain evidence.",
            )
        )
    return tuple(issues)


