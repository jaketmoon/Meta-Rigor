from __future__ import annotations

import hashlib
import json
from pathlib import Path
from metarigor.data_extraction.benchmark_v3.models import ReviewBenchmarkInventory
from metarigor.evidence_certainty.models import GRADE_DOMAINS
from .models import BenchmarkCatalog, BenchmarkGold
_REPO_ROOT = Path(__file__).resolve().parents[5]


_DATASET_ROOT = Path(__file__).with_name("dataset")


_DE_INVENTORY_PATH = (
    _REPO_ROOT / "backend/src/metarigor/data_extraction/benchmark_v3/dataset/inventory.json"
)


CATALOG_SHA256 = "19b22ce0aac3f1fd9c7e68f4de5e32d71127889122a07c9a338585066d59ef97"


GOLD_SHA256 = "f513e9ca8ac9464b34e6e3e7c1c5eb1b4b4aabcf653d2fb3b36c89b45e38bdc5"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark asset must be a JSON object: {path}")
    return payload


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


def _load_de_inventory() -> ReviewBenchmarkInventory:
    return ReviewBenchmarkInventory.model_validate(_read_json(_DE_INVENTORY_PATH))


def load_benchmark_catalog() -> BenchmarkCatalog:
    """只加载候选可见输入；此函数绝不读取 gold 或 external comparator。"""

    catalog_path = _DATASET_ROOT / "cases.json"
    if _sha256(catalog_path.read_bytes()) != CATALOG_SHA256:
        raise ValueError("pinned EC V1 candidate catalog changed")
    catalog = BenchmarkCatalog.model_validate(_read_json(catalog_path))
    inventory_content = _DE_INVENTORY_PATH.read_bytes()
    if _sha256(inventory_content) != catalog.data_extraction_inventory_sha256:
        raise ValueError("pinned Data Extraction V3 inventory changed")
    inventory = _load_de_inventory()
    inventory_by_id = {item.case_id: item for item in inventory.cases}
    expected_cases = {
        item.case_id
        for item in inventory.cases
        if item.candidate_request_path and item.reference_package_path and item.checkpoint_path
    }
    actual_cases = {item.case_id for item in catalog.cases}
    if actual_cases != expected_cases:
        raise ValueError("EC V1 must project exactly the ten runnable DE V3 Review cases")
    for case in catalog.cases:
        shell = inventory_by_id[case.case_id]
        if shell.legacy_case_id != case.legacy_case_id:
            raise ValueError(f"legacy case binding changed: {case.case_id}")
        analyses = {item.analysis_id for item in shell.analyses}
        if case.selected_analysis_id not in analyses:
            raise ValueError(f"selected analysis is absent from DE V3: {case.case_id}")
        protocol = _resolve_repo_asset(case.protocol_asset_path)
        if _sha256(protocol.read_bytes()) != case.protocol_asset_sha256:
            raise ValueError(f"accepted Protocol hash changed: {case.case_id}")
        documents = {item.document_id: item for item in case.source_documents}
        for document in case.source_documents:
            path = _resolve_repo_asset(document.path)
            if _sha256(path.read_bytes()) != document.sha256:
                raise ValueError(
                    f"Source Document hash changed: {case.case_id}/{document.document_id}"
                )
        for fact in case.evidence_facts:
            document = documents[fact.document_id]
            text = _resolve_repo_asset(document.path).read_bytes().decode("utf-8")
            if text.count(fact.quote) != 1:
                raise ValueError(
                    f"Evidence Fact quote must resolve exactly once: {case.case_id}/{fact.fact_id}"
                )
        if case.source_closure_document_id is not None:
            document = documents[case.source_closure_document_id]
            text = _resolve_repo_asset(document.path).read_bytes().decode("utf-8")
            if text.count(case.source_closure_quote) != 1:
                raise ValueError(f"scope evidence quote must resolve exactly once: {case.case_id}")
        for scope_evidence in case.scope_basis_evidence:
            scope_document = documents[scope_evidence.document_id]
            scope_text = _resolve_repo_asset(scope_document.path).read_bytes().decode("utf-8")
            if scope_text.count(scope_evidence.quote) != 1:
                raise ValueError(f"scope basis quote must resolve exactly once: {case.case_id}")
        for basis in case.study_design_bases:
            for design_evidence in basis.evidence:
                document = documents[design_evidence.document_id]
                text = _resolve_repo_asset(document.path).read_bytes().decode("utf-8")
                if text.count(design_evidence.quote) != 1:
                    raise ValueError(
                        "Study design quote must resolve exactly once: "
                        f"{case.case_id}/{basis.study_id}"
                    )
    return catalog


def load_benchmark_gold(catalog: BenchmarkCatalog | None = None) -> BenchmarkGold:
    path = _DATASET_ROOT / "gold.json"
    if _sha256(path.read_bytes()) != GOLD_SHA256:
        raise ValueError("pinned EC V1 gold changed")
    reference = BenchmarkGold.model_validate(_read_json(path))
    candidate_catalog = catalog or load_benchmark_catalog()
    if reference.candidate_catalog_sha256 != CATALOG_SHA256:
        raise ValueError("EC V1 gold does not bind the current candidate catalog")
    candidates = {item.case_id: item for item in candidate_catalog.cases}
    if {item.case_id for item in reference.cases} != set(candidates):
        raise ValueError("gold cases do not close candidate catalog")
    for gold in reference.cases:
        candidate = candidates[gold.case_id]
        if gold.selected_analysis_id != candidate.selected_analysis_id:
            raise ValueError(f"gold selected analysis differs: {gold.case_id}")
        if gold.disposition == "SCORED" and {item.domain for item in gold.domains} != set(
            GRADE_DOMAINS
        ):
            raise ValueError(f"gold does not contain the five frozen domains: {gold.case_id}")
        known_facts = {item.fact_id for item in candidate.evidence_facts}
        fact_domains = {
            item.fact_id: set(item.domain_relevance) for item in candidate.evidence_facts
        }
        if any(
            fact_id not in known_facts
            for domain in gold.domains
            for fact_id in domain.evidence_fact_ids
        ):
            raise ValueError(f"gold cites a fact outside the candidate closure: {gold.case_id}")
        if any(
            domain.domain not in fact_domains[fact_id]
            for domain in gold.domains
            for fact_id in domain.evidence_fact_ids
        ):
            raise ValueError(f"gold cites a fact outside its domain: {gold.case_id}")
    return reference


