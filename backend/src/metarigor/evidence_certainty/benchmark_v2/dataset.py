from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from metarigor.evidence_certainty.benchmark_v1 import load_benchmark_catalog as load_v1_catalog
from metarigor.evidence_certainty.benchmark_v1 import load_benchmark_gold as load_v1_gold
from metarigor.evidence_certainty.benchmark_v1.dataset import CATALOG_SHA256 as V1_CATALOG_SHA256
from metarigor.evidence_certainty.benchmark_v1.dataset import GOLD_SHA256 as V1_GOLD_SHA256
from .freeze import FREEZER_ID, FREEZER_VERSION, candidate_exposes_grade_disclosure
from .models import BenchmarkV2Catalog, BenchmarkV2Gold, ExternalComparatorDocument, ReviewBenchmarkCase, ReviewCandidateDocument, ReviewFigureEvidence, ReviewGoldCase, ReviewGoldDomain, ReviewLevelSufficiencyContract, ReviewTextEvidence, V2CandidateSeal
_REPO_ROOT = Path(__file__).resolve().parents[5]


_DATASET_ROOT = Path(__file__).with_name("dataset")


CATALOG_SHA256 = "021b4128b042c485ddf853ee9610ce18afaa2b3b8b83d59eb3f436791af29ae8"


GOLD_SHA256 = "4bee294be1d78c2218efc9163f4a506e9e6e63a77d34e17fb2bab90c57b92eaf"


REVIEW_SUFFICIENCY_SHA256 = "27c1b55d19ad807878f7cca203d00770c37c601669482e99d0bc5ad1a94012d3"


_REVIEW_SUFFICIENCY_PATH = _DATASET_ROOT / "review-level-sufficiency-v3.json"


_NUMBER = re.compile(r"(?<![A-Za-z0-9.])(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?")


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_url_matches_pmcid(source_url: str, pmcid: str) -> bool:
    numeric_id = pmcid.removeprefix("PMC")
    return (
        f"/{numeric_id}/" in source_url
        if "cdn.ncbi.nlm.nih.gov" in source_url
        else f"/{pmcid}." in source_url
    )


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark asset must be a JSON object: {path}")
    return payload


def _bound_numbers(value: str) -> tuple[float, ...]:
    numeric = [float(item.replace(",", "")) for item in _NUMBER.findall(value)]
    lowered = value.casefold()
    numeric.extend(
        float(number)
        for word, number in _NUMBER_WORDS.items()
        if re.search(rf"\b{word}\b", lowered)
    )
    return tuple(numeric)


def _contains_number(numbers: tuple[float, ...], expected: float) -> bool:
    return any(abs(item - expected) < 1e-9 for item in numbers)


def _contains_integer_or_arm_sum(numbers: tuple[float, ...], expected: int) -> bool:
    integer_values = [int(item) for item in numbers if item.is_integer()]
    if expected in integer_values:
        return True
    return any(
        first + second == expected
        for index, first in enumerate(integer_values)
        for second in integer_values[index + 1 :]
    )


def _effect_tuples(value: str, measure: str) -> tuple[tuple[float, float, float], ...]:
    measure_patterns = {
        "RR": (
            (r"\bRR\b", 0),
            (r"\b(?:risk\s+ratios?|relative\s+risks?)\b", re.IGNORECASE),
        ),
        "OR": ((r"\bOR\b", 0), (r"\bodds\s+ratios?\b", re.IGNORECASE)),
    }[measure]
    starts = sorted(
        {
            match.start()
            for pattern, flags in measure_patterns
            for match in re.finditer(pattern, value, flags)
        }
    )
    matches: list[tuple[float, float, float]] = []
    for start in starts:
        segment = value[start : start + 160]
        if re.search(r"\brange\b", segment[:40], re.IGNORECASE):
            continue
        if re.search(r"\bCI\b|confidence\s+interval", segment, re.IGNORECASE) is None:
            continue
        segment = re.sub(
            r"95\s*%\s*(?:confidence\s+interval|CI)",
            "",
            segment,
            flags=re.IGNORECASE,
        )
        numbers = _bound_numbers(segment)
        if len(numbers) < 3:
            continue
        candidate = (numbers[0], numbers[1], numbers[2])
        if not matches or matches[-1] != candidate:
            matches.append(candidate)
    return tuple(matches)


def _validate_effect_binding(case: ReviewBenchmarkCase, lines: list[str]) -> None:
    anchors = {item.anchor_id: item for item in case.anchors}
    bound_text = "\n".join(
        lines[anchors[anchor_id].line_number - 1] for anchor_id in case.source_bindings.effect
    )
    tuple_anchor = anchors[case.source_bindings.effect_tuple_anchor_id]
    tuple_text = lines[tuple_anchor.line_number - 1]
    tuples = _effect_tuples(tuple_text, case.effect.measure)
    ordinal = case.source_bindings.effect_tuple_ordinal
    expected_tuple = (
        case.effect.estimate,
        case.effect.ci_lower,
        case.effect.ci_upper,
    )
    if ordinal >= len(tuples) or tuples[ordinal] != expected_tuple:
        raise ValueError(f"effect binding does not support selected tuple: {case.case_id}")
    numbers = _bound_numbers(bound_text)
    if not _contains_integer_or_arm_sum(numbers, case.effect.study_count):
        raise ValueError(f"effect binding does not support study_count: {case.case_id}")
    for field_name in ("participant_count", "event_count"):
        expected = getattr(case.effect, field_name)
        if expected is not None and not _contains_integer_or_arm_sum(numbers, expected):
            raise ValueError(f"effect binding does not support {field_name}: {case.case_id}")
    if case.effect.i2_percent is not None and not _contains_number(numbers, case.effect.i2_percent):
        raise ValueError(f"effect binding does not support i2_percent: {case.case_id}")


def _resolve_repo_asset(relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("benchmark asset path must be repository-relative")
    resolved = (_REPO_ROOT / candidate).resolve()
    try:
        resolved.relative_to(_REPO_ROOT)
    except ValueError as error:
        raise ValueError("benchmark asset path escapes repository") from error
    if not resolved.is_file():
        raise ValueError(f"benchmark asset is missing: {relative}")
    return resolved


def _resolve_dataset_asset(relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("frozen dataset path must be relative")
    resolved = (_DATASET_ROOT / candidate).resolve()
    try:
        resolved.relative_to(_DATASET_ROOT)
    except ValueError as error:
        raise ValueError("frozen dataset path escapes dataset root") from error
    if not resolved.is_file():
        raise ValueError(f"frozen dataset asset is missing: {relative}")
    return resolved


def _manifest(case: ReviewBenchmarkCase) -> tuple[dict, Path]:
    manifest_path = _resolve_repo_asset(case.source.manifest_path)
    content = manifest_path.read_bytes()
    if _sha256(content) != case.source.manifest_sha256:
        raise ValueError(f"frozen manifest changed: {case.case_id}")
    manifest = _read_json(manifest_path)
    freezer = manifest.get("freezer")
    article = manifest.get("article")
    if freezer != {"id": FREEZER_ID, "version": FREEZER_VERSION}:
        raise ValueError(f"unexpected freezer binding: {case.case_id}")
    if not isinstance(article, dict) or article.get("pmcid") != case.source.pmcid:
        raise ValueError(f"frozen PMCID binding changed: {case.case_id}")
    license_text = article.get("license_text")
    license_url = article.get("license_url")
    license_binding = f"{license_text or ''} {license_url or ''}".lower()
    is_attribution = (
        "attribution" in license_binding
        or "cc by" in license_binding
        or "creativecommons.org/licenses/by/" in license_binding
        or "creativecommons.org/licenses/by-sa/" in license_binding
    )
    is_restricted = any(
        marker in license_binding
        for marker in ("/by-nc", "noncommercial", "non-commercial", "/by-nd")
    )
    if not isinstance(license_text, str) or not is_attribution or is_restricted:
        raise ValueError(f"frozen source is outside the CC BY gate: {case.case_id}")
    return manifest, manifest_path


def _validated_candidate(case) -> None:
    manifest, _ = _manifest(case)
    raw = manifest.get("raw_source")
    candidate = manifest.get("candidate")
    comparator = manifest.get("external_comparator")
    if not all(isinstance(item, dict) for item in (raw, candidate, comparator)):
        raise ValueError(f"frozen manifest artifact binding is incomplete: {case.case_id}")
    raw_path = _resolve_dataset_asset(raw["path"])
    candidate_path = _resolve_dataset_asset(candidate["path"])
    if _sha256(raw_path.read_bytes()) != raw.get("sha256"):
        raise ValueError(f"raw PMC JATS changed: {case.case_id}")
    candidate_content = candidate_path.read_bytes()
    if _sha256(candidate_content) != candidate.get("sha256"):
        raise ValueError(f"frozen candidate changed: {case.case_id}")
    text = candidate_content.decode("utf-8")
    if candidate_exposes_grade_disclosure(text):
        raise ValueError(f"frozen candidate exposes GRADE disclosure: {case.case_id}")
    lines = text.splitlines()
    if len(lines) != candidate.get("line_count"):
        raise ValueError(f"frozen candidate line count changed: {case.case_id}")
    for anchor in case.anchors:
        if anchor.line_number > len(lines):
            raise ValueError(f"candidate anchor is outside document: {case.case_id}")
        line = lines[anchor.line_number - 1]
        if _sha256(line.encode("utf-8")) != anchor.line_sha256:
            raise ValueError(f"candidate anchor changed: {case.case_id}/{anchor.anchor_id}")
    _validate_effect_binding(case, lines)


def load_benchmark_catalog() -> BenchmarkV2Catalog:
    """Load the 20-case candidate catalog without reading gold or comparator text."""

    path = _DATASET_ROOT / "cases.json"
    if _sha256(path.read_bytes()) != CATALOG_SHA256:
        raise ValueError("pinned EC V2 candidate catalog changed")
    catalog = BenchmarkV2Catalog.model_validate(_read_json(path))
    if catalog.inherited_v1_catalog_sha256 != V1_CATALOG_SHA256:
        raise ValueError("EC V2 inherited catalog hash differs from pinned EC V1")
    v1_catalog = load_v1_catalog()
    inherited_v1_ids = set(catalog.inherited_v1_scored_case_ids) | set(
        catalog.retained_v1_abstention_case_ids
    )
    if inherited_v1_ids != {item.case_id for item in v1_catalog.cases}:
        raise ValueError("EC V2 inherited identities differ from pinned EC V1")
    for case in catalog.review_cases:
        _validated_candidate(case)
    return catalog


def load_review_level_sufficiency_contract(
    catalog: BenchmarkV2Catalog | None = None,
) -> ReviewLevelSufficiencyContract:
    """Load V2 PMC Review-level sufficiency without reading gold or comparators."""

    catalog = catalog or load_benchmark_catalog()
    path = _REVIEW_SUFFICIENCY_PATH
    if _sha256(path.read_bytes()) != REVIEW_SUFFICIENCY_SHA256:
        raise ValueError("pinned EC V2 review sufficiency contract changed")
    contract = ReviewLevelSufficiencyContract.model_validate(_read_json(path))
    expected_ids = tuple(item.case_id for item in catalog.review_cases)
    observed_ids = tuple(item.case_id for item in contract.cases)
    if observed_ids != expected_ids:
        raise ValueError("review sufficiency case order differs from V2 catalog")
    catalog_cases = {item.case_id: item for item in catalog.review_cases}
    for case in contract.cases:
        catalog_case = catalog_cases[case.case_id]
        manifest, _ = _manifest(catalog_case)
        candidate_binding = manifest["candidate"]
        candidate_path = _resolve_dataset_asset(candidate_binding["path"])
        candidate_content = candidate_path.read_bytes()
        candidate_lines = candidate_content.decode("utf-8").splitlines()
        for domain in case.domains:
            evidence_ids = [item.evidence_id for item in domain.supplemental_evidence]
            if len(evidence_ids) != len(set(evidence_ids)):
                raise ValueError(
                    "review sufficiency evidence ids are duplicated: "
                    f"{case.case_id}/{domain.domain}"
                )
            for evidence in domain.supplemental_evidence:
                if evidence.source_pmcid != catalog_case.source.pmcid:
                    raise ValueError(
                        f"review sufficiency PMCID differs from catalog: "
                        f"{case.case_id}/{evidence.evidence_id}"
                    )
                if isinstance(evidence, ReviewTextEvidence):
                    if (
                        evidence.candidate_path != candidate_binding["path"]
                        or evidence.candidate_sha256 != candidate_binding["sha256"]
                        or _sha256(candidate_content) != evidence.candidate_sha256
                    ):
                        raise ValueError(
                            f"review text evidence candidate binding changed: "
                            f"{case.case_id}/{evidence.evidence_id}"
                        )
                    if evidence.line_number > len(candidate_lines):
                        raise ValueError(
                            f"review text evidence line is outside candidate: "
                            f"{case.case_id}/{evidence.evidence_id}"
                        )
                    quote = candidate_lines[evidence.line_number - 1]
                    if (
                        quote != evidence.quote
                        or _sha256(quote.encode("utf-8")) != evidence.line_sha256
                    ):
                        raise ValueError(
                            f"review text evidence quote changed: "
                            f"{case.case_id}/{evidence.evidence_id}"
                        )
                    continue
                if not isinstance(evidence, ReviewFigureEvidence):
                    raise TypeError("unsupported review sufficiency evidence representation")
                if not Path(evidence.artifact_path).name.startswith(evidence.source_pmcid):
                    raise ValueError(
                        f"review figure filename is not source-bound: "
                        f"{case.case_id}/{evidence.evidence_id}"
                    )
                if not _source_url_matches_pmcid(evidence.source_url, evidence.source_pmcid):
                    raise ValueError(
                        f"review figure URL is not source-bound: "
                        f"{case.case_id}/{evidence.evidence_id}"
                    )
                artifact = _resolve_dataset_asset(evidence.artifact_path)
                if _sha256(artifact.read_bytes()) != evidence.artifact_sha256:
                    raise ValueError(
                        f"review sufficiency figure changed: {case.case_id}/{evidence.evidence_id}"
                    )
                if evidence.source_document_path is not None:
                    source_document = _resolve_dataset_asset(evidence.source_document_path)
                    if not source_document.name.startswith(evidence.source_pmcid):
                        raise ValueError(
                            f"review figure source document is not source-bound: "
                            f"{case.case_id}/{evidence.evidence_id}"
                        )
                    if _sha256(source_document.read_bytes()) != evidence.source_document_sha256:
                        raise ValueError(
                            f"review figure source document changed: "
                            f"{case.case_id}/{evidence.evidence_id}"
                        )
            resolution = domain.source_resolution
            if resolution is None:
                continue
            if resolution.source_pmcid != catalog_case.source.pmcid:
                raise ValueError(f"review source resolution PMCID differs: {case.case_id}")
            if not Path(resolution.selected_artifact_path).name.startswith(
                resolution.source_pmcid
            ) or not _source_url_matches_pmcid(
                resolution.selected_source_url,
                resolution.source_pmcid,
            ):
                raise ValueError(f"review selected source is not source-bound: {case.case_id}")
            selected = _resolve_dataset_asset(resolution.selected_artifact_path)
            if _sha256(selected.read_bytes()) != resolution.selected_artifact_sha256:
                raise ValueError(f"review selected source changed: {case.case_id}")
            if (
                catalog_case.effect.study_count != resolution.selected_study_count
                or catalog_case.effect.participant_count
                != resolution.selected_intervention_total + resolution.selected_control_total
            ):
                raise ValueError(f"review selected source differs from catalog: {case.case_id}")
            for path_value, expected_sha in (
                (resolution.excluded_artifact_path, resolution.excluded_artifact_sha256),
                (
                    resolution.excluded_source_document_path,
                    resolution.excluded_source_document_sha256,
                ),
            ):
                if not Path(path_value).name.startswith(resolution.source_pmcid):
                    raise ValueError(
                        f"review excluded source is not source-bound: {case.case_id}"
                    )
                excluded = _resolve_dataset_asset(path_value)
                if _sha256(excluded.read_bytes()) != expected_sha:
                    raise ValueError(f"review excluded source changed: {case.case_id}")
    return contract


def review_sufficiency_contract_bytes() -> bytes:
    """Return hash-verified original V2 contract text for sealing in the Run Folder."""

    content = _REVIEW_SUFFICIENCY_PATH.read_bytes()
    if _sha256(content) != REVIEW_SUFFICIENCY_SHA256:
        raise ValueError("pinned EC V2 review sufficiency contract changed")
    return content


def review_sufficiency_artifact_bytes(
    contract: ReviewLevelSufficiencyContract,
) -> tuple[tuple[str, bytes, str], ...]:
    """Return figure/source-document bytes actually referenced by the contract, stably sorted by path."""

    bindings: dict[str, tuple[bytes, str]] = {}
    for case in contract.cases:
        for domain in case.domains:
            for evidence in domain.supplemental_evidence:
                if not isinstance(evidence, ReviewFigureEvidence):
                    continue
                for path_value, expected_sha in (
                    (evidence.artifact_path, evidence.artifact_sha256),
                    (evidence.source_document_path, evidence.source_document_sha256),
                ):
                    if path_value is None or expected_sha is None:
                        continue
                    path = _resolve_dataset_asset(path_value)
                    content = path.read_bytes()
                    if _sha256(content) != expected_sha:
                        raise ValueError(f"review sufficiency artifact changed: {path_value}")
                    bindings[path_value] = (content, expected_sha)
            resolution = domain.source_resolution
            if resolution is not None:
                for path_value, expected_sha in (
                    (resolution.selected_artifact_path, resolution.selected_artifact_sha256),
                    (resolution.excluded_artifact_path, resolution.excluded_artifact_sha256),
                    (
                        resolution.excluded_source_document_path,
                        resolution.excluded_source_document_sha256,
                    ),
                ):
                    path = _resolve_dataset_asset(path_value)
                    content = path.read_bytes()
                    if _sha256(content) != expected_sha:
                        raise ValueError(
                            f"review sufficiency source-resolution artifact changed: {path_value}"
                        )
                    bindings[path_value] = (content, expected_sha)
    return tuple(
        (path, bindings[path][0], bindings[path][1]) for path in sorted(bindings)
    )


def load_benchmark_gold(catalog: BenchmarkV2Catalog | None = None) -> BenchmarkV2Gold:
    """Load independently curated gold for 13 review-level cases."""

    candidate_catalog = catalog or load_benchmark_catalog()
    path = _DATASET_ROOT / "gold.json"
    if _sha256(path.read_bytes()) != GOLD_SHA256:
        raise ValueError("pinned EC V2 review gold changed")
    gold = BenchmarkV2Gold.model_validate(_read_json(path))
    if gold.candidate_catalog_sha256 != CATALOG_SHA256:
        raise ValueError("EC V2 gold does not bind the current candidate catalog")
    if gold.inherited_v1_gold_sha256 != V1_GOLD_SHA256:
        raise ValueError("EC V2 gold does not bind the pinned V1 gold")
    v1_gold_path = _REPO_ROOT / (
        "backend/src/metarigor/evidence_certainty/benchmark_v1/dataset/gold.json"
    )
    if _sha256(v1_gold_path.read_bytes()) != gold.inherited_v1_gold_sha256:
        raise ValueError("EC V2 inherited V1 gold binding changed")
    v1_gold = load_v1_gold()
    scored = tuple(item.case_id for item in v1_gold.cases if item.disposition == "SCORED")
    abstained = tuple(
        item.case_id
        for item in v1_gold.cases
        if item.disposition == "CORRECT_ABSTENTION_SOURCE_NOT_CLOSED"
    )
    if candidate_catalog.inherited_v1_scored_case_ids != scored:
        raise ValueError("EC V2 inherited scored cases differ from pinned EC V1 gold")
    if candidate_catalog.retained_v1_abstention_case_ids != abstained:
        raise ValueError("EC V2 retained abstentions differ from pinned EC V1 gold")
    candidates = {item.case_id: item for item in candidate_catalog.review_cases}
    if {item.case_id for item in gold.cases} != set(candidates):
        raise ValueError("EC V2 review gold does not close the candidate catalog")
    for reference in gold.cases:
        candidate = candidates[reference.case_id]
        if reference.selected_analysis_id != candidate.selected_analysis_id:
            raise ValueError(f"selected analysis differs: {reference.case_id}")
        anchors = {item.anchor_id: item for item in candidate.anchors}
        for domain in reference.domains:
            if any(anchor_id not in anchors for anchor_id in domain.evidence_anchor_ids):
                raise ValueError(f"gold references unknown anchor: {reference.case_id}")
            if any(
                domain.domain not in anchors[anchor_id].domain_relevance
                for anchor_id in domain.evidence_anchor_ids
            ):
                raise ValueError(f"gold anchor has wrong domain: {reference.case_id}")
    return gold


def load_benchmark_review_candidates(
    catalog: BenchmarkV2Catalog | None = None,
) -> tuple[ReviewCandidateDocument, ...]:
    """Materialize text for the 13 new Review candidates without reading gold or comparators."""

    candidate_catalog = catalog or load_benchmark_catalog()
    documents = []
    for case in candidate_catalog.review_cases:
        manifest, _ = _manifest(case)
        binding = manifest["candidate"]
        path = _resolve_dataset_asset(binding["path"])
        content = path.read_bytes()
        if _sha256(content) != binding["sha256"]:
            raise ValueError(f"frozen candidate changed: {case.case_id}")
        documents.append(
            ReviewCandidateDocument(
                case_id=case.case_id,
                pmcid=case.source.pmcid,
                path=path.relative_to(_REPO_ROOT).as_posix(),
                sha256=binding["sha256"],
                content=content.decode("utf-8"),
            )
        )
    return tuple(documents)


def load_combined_gold(
    catalog: BenchmarkV2Catalog | None = None,
) -> tuple[ReviewGoldCase, ...]:
    """Return a unified 20-case view of 7 V1 scored gold cases and 13 V2 review gold cases."""

    candidate_catalog = catalog or load_benchmark_catalog()
    review_gold = load_benchmark_gold(candidate_catalog)
    v1_gold = load_v1_gold()
    inherited = []
    wanted = set(candidate_catalog.inherited_v1_scored_case_ids)
    for case in v1_gold.cases:
        if case.case_id not in wanted:
            continue
        inherited.append(
            ReviewGoldCase(
                case_id=case.case_id,
                selected_analysis_id=case.selected_analysis_id,
                domains=tuple(
                    ReviewGoldDomain(
                        domain=domain.domain,
                        judgment=domain.judgment,
                        rationale=domain.rationale,
                        evidence_anchor_ids=domain.evidence_fact_ids,
                    )
                    for domain in case.domains
                ),
                overall_downgrade_levels=case.overall_downgrade_levels,
                final_certainty=case.final_certainty,
                overall_rationale=case.overall_rationale,
                curator_confidence=case.curator_confidence,
            )
        )
    combined = tuple(inherited) + review_gold.cases
    if tuple(item.case_id for item in combined) != candidate_catalog.scored_case_ids:
        raise ValueError("combined EC V2 gold order differs from the scored catalog")
    if sum(len(item.domains) for item in combined) != 100:
        raise ValueError("combined EC V2 gold must contain exactly 100 domain judgments")
    return combined


def load_benchmark_external_comparators(
    seal: V2CandidateSeal,
    catalog: BenchmarkV2Catalog | None = None,
) -> tuple[ExternalComparatorDocument, ...]:
    """Read comparator text for 13 Reviews only after all 20 candidates have been sealed."""

    candidate_catalog = catalog or load_benchmark_catalog()
    if seal.catalog_sha256 != candidate_catalog.canonical_sha256:
        raise ValueError("candidate seal does not bind the EC V2 catalog")
    sealed_case_ids = tuple(item.case_id for item in seal.prediction_set.cases)
    if sealed_case_ids != candidate_catalog.scored_case_ids:
        raise ValueError("external comparator requires ordered predictions for all 20 cases")
    documents = []
    for case in candidate_catalog.review_cases:
        manifest, _ = _manifest(case)
        binding = manifest["external_comparator"]
        path = _resolve_dataset_asset(binding["path"])
        content = path.read_bytes()
        if _sha256(content) != binding["sha256"]:
            raise ValueError(f"external comparator changed: {case.case_id}")
        documents.append(
            ExternalComparatorDocument(
                case_id=case.case_id,
                pmcid=case.source.pmcid,
                path=path.relative_to(_REPO_ROOT).as_posix(),
                sha256=binding["sha256"],
                content=content.decode("utf-8"),
            )
        )
    return tuple(documents)

