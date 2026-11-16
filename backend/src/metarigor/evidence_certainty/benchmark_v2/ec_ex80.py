"""Run the separately sealed EC-ex80 cohort with the existing EC V2 engine."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import get_args
from uuid import uuid4

from metarigor.evidence_certainty.definitions import domain_template, overall_template
from metarigor.evidence_certainty.models import DomainJudgment, GRADE_DOMAINS, canonical_json
from metarigor.local_run import RunFolder, RunManifest, RunStatus, StageRunner
from metarigor.local_run.runner import utc_now
from metarigor.research import RESEARCH_ROOT

from . import live, task_contract_ablation
from .models import BenchmarkV2EvidenceAnchorSidecar, ReviewCaseSufficiency
from .source_evidence import _verify_sidecar

BENCHMARK = "ec-ex80"
DEFAULT_DATA = Path("output/ec-ex80-data")


def catalog() -> dict:
    return json.loads((RESEARCH_ROOT / "inputs/ec-ex80/catalog.json").read_bytes())


def case_ids() -> tuple[str, ...]:
    cohorts = catalog()["cohorts"]
    ids = tuple(cid for key in ("original_20", "added_60") for cid in cohorts[key]["case_ids"])
    if len(ids) != 80 or len(set(ids)) != 80:
        raise ValueError("EC-ex80 catalog must contain 80 unique cases")
    return ids


def load_bundle(root: Path) -> tuple[RunFolder, dict]:
    folder = RunFolder(root)
    plan = json.loads(folder.read_verified("plan.json", catalog()["prepared_plan_sha256"]))
    if tuple(item["case_id"] for item in plan["prepared"]) != case_ids() or plan["gold_loaded"]:
        raise ValueError("EC-ex80 bundle differs from the frozen candidate inventory")
    return folder, plan


def read_candidate(source: RunFolder, entry: dict) -> live.V2LiveCandidateInput:
    candidate = live.V2LiveCandidateInput.model_validate_json(
        source.read_verified(entry["input_path"], entry["input_sha256"])
    )
    if candidate.case_id != entry["case_id"]:
        raise ValueError("EC-ex80 candidate identity differs from its plan")
    return candidate


def prepare_cases(
    source: RunFolder, plan: dict, selected_case: str, *, target: RunFolder | None = None,
) -> tuple[live._PreparedCandidate, ...]:
    """Validate quotes and raw provenance and optionally copy them into a fresh Run."""
    contract_bytes = source.read_verified(plan["contract_path"], plan["contract_sha256"])
    contract = json.loads(contract_bytes)
    if contract["contract_id"] != plan["contract_id"]:
        raise ValueError("EC-ex80 sufficiency contract identity differs from its plan")
    sufficiency = {row["case_id"]: ReviewCaseSufficiency.model_validate(row) for row in contract["cases"]}
    if set(sufficiency) != set(case_ids()) or len(contract["cases"]) != len(sufficiency):
        raise ValueError("EC-ex80 sufficiency contract case inventory differs from its plan")
    if target is not None:
        target.write_immutable(plan["contract_path"], contract_bytes)

    prepared = []
    copied_sources = set()
    for entry in plan["prepared"]:
        cid = entry["case_id"]
        if selected_case not in ("all", cid):
            continue
        candidate = read_candidate(source, entry)
        sidecar_bytes = source.read_verified(entry["sidecar_path"], entry["sidecar_sha256"])
        sidecars = tuple(BenchmarkV2EvidenceAnchorSidecar.model_validate(row) for row in json.loads(sidecar_bytes))
        expected = {item.evidence_id: (item.quote, item.domain_relevance) for item in candidate.evidence}
        observed = {}
        for sidecar in sidecars:
            span = sidecar.source_span
            if span is None:
                raise ValueError(f"EC-ex80 evidence has no SourceSpan: {cid}/{sidecar.evidence_id}")
            observed[sidecar.evidence_id] = (span.quote, sidecar.domain_relevance)
            issues = _verify_sidecar(sidecar, folder=source, case_id=cid, domain=sidecar.domain_relevance[0])
            if issues:
                raise ValueError(f"EC-ex80 source verification failed: {cid}/{sidecar.evidence_id}: {issues[0].detail}")
            if target is not None:
                bind_sources = [(span.input_path, span.document_sha256)]
                bind_sources.extend((raw.input_path, raw.document_sha256) for raw in sidecar.raw_source_provenance)
                for path, digest in bind_sources:
                    if (path, digest) not in copied_sources:
                        target.write_immutable(path, source.read_verified(path, digest))
                        copied_sources.add((path, digest))
        if observed != expected or len(observed) != len(sidecars):
            raise ValueError(f"EC-ex80 candidate evidence differs from its sidecars: {cid}")
        if target is not None:
            target.write_immutable(entry["input_path"], source.read_verified(entry["input_path"], entry["input_sha256"]))
            target.write_immutable(entry["sidecar_path"], sidecar_bytes)
        prepared.append(live._PreparedCandidate(
            case_id=cid,
            input=candidate,
            review_sufficiency=sufficiency[cid],
            evidence_sidecars=sidecars,
            evidence_sidecar_path=entry["sidecar_path"],
            evidence_sidecar_sha256=entry["sidecar_sha256"],
            review_sufficiency_contract=(plan["contract_id"], plan["contract_sha256"]),
        ))

    if not prepared:
        raise ValueError(f"Unknown EC-ex80 case: {selected_case}")
    return tuple(prepared)


def write_cli_packages(root: Path, data_root: Path, selected_case: str) -> Path:
    source, plan = load_bundle(data_root)
    folder = RunFolder.create(root)
    for entry in plan["prepared"]:
        if selected_case not in ("all", entry["case_id"]):
            continue
        candidate = read_candidate(source, entry)
        base = candidate.case_id
        payload = source.read_verified(entry["input_path"], entry["input_sha256"])
        artifact = folder.write_immutable(f"{base}/input.json", payload)
        folder.write_immutable(f"{base}/input-manifest.json", canonical_json({
            "benchmark": BENCHMARK,
            "case_id": base,
            "input_json_path": "input.json",
            "input_json_sha256": artifact.sha256,
            "accessible_files": ["input-manifest.json", "input.json"],
            "reference_answers_loaded": False,
        }))
    return folder.root


def _ablation_request(candidate: live.V2LiveCandidateInput) -> tuple[dict, list[dict]]:
    """Project the same upstream evidence onto the source-only task-contract ablation."""
    question = (
        "Please assess the certainty of the randomized evidence using Core GRADE 2025 for "
        f"the following research question. Population: {candidate.population} "
        f"Intervention: {candidate.intervention} Comparator: {candidate.comparator} "
        f"Outcome: {candidate.outcome} Follow-up: {candidate.timepoint} "
        "Explain the assessment and provide verifiable sources, including any inability to assess."
    )
    quotes: dict[str, list[str]] = {}
    for evidence in candidate.evidence:
        quotes.setdefault(evidence.quote, []).append(evidence.evidence_id)
    sources = [{"source_id": "S000", "quote": candidate.candidate_document, "original_ids": []}]
    for index, quote in enumerate(sorted(quotes, key=lambda text: sha256(text.encode()).hexdigest()), 1):
        sources.append({"source_id": f"S{index:03d}", "quote": quote, "original_ids": sorted(set(quotes[quote]))})
    materials = [f"SOURCE {row['source_id']}\n{row['quote']}" for row in sources]
    materials.append(
        "Existing upstream numerical synthesis summary (curated, not a new calculation):\n"
        + json.dumps(candidate.effect.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    )
    return {"request": question, "materials": "\n\n".join(materials)}, sources


async def run(*, data_root, output, selected_case, condition, runtime, timeout_seconds):
    source, plan = load_bundle(data_root)
    folder = RunFolder(output)
    prepared = prepare_cases(source, plan, selected_case, target=folder)
    case_set = {case.case_id for case in prepared}
    skills = {}
    if condition == "full":
        skills.update({d: domain_template(d).skill_text or "" for d in GRADE_DOMAINS if d != "IMPRECISION"})
        skills.update(review=live._review_sufficiency_skill_text(), overall=overall_template().skill_text or "")
    elif condition == "no-task-contract":
        skills["methods"] = (RESEARCH_ROOT / "inputs/ec/methods.txt").read_text(encoding="utf-8")
    skill_bindings = {
        name: asdict(folder.write_immutable(f"execution/skills/{name}.txt", text.encode("utf-8")))
        for name, text in skills.items()
    }
    binding = folder.write_immutable("plan.json", canonical_json({
        "benchmark": BENCHMARK,
        "condition": condition,
        "runtime": runtime.binding("worker"),
        "prepared_plan_sha256": catalog()["prepared_plan_sha256"],
        "prepared": [row for row in plan["prepared"] if row["case_id"] in case_set],
        "contract_path": plan["contract_path"],
        "contract_sha256": plan["contract_sha256"],
        "skills": skill_bindings,
        "gold_loaded": False,
    }))
    run_id = str(uuid4())
    manifest = RunManifest(folder.manifest_path)
    manifest.initialize()
    manifest.create_run(run_id=run_id, specialist="evidence_certainty", input_path=binding.path,
                        input_sha256=binding.sha256, started_at=utc_now())
    manifest.set_run_status(run_id, RunStatus.RUNNING)
    stages = StageRunner(run_id=run_id, folder=folder, manifest=manifest)
    model_limiter = asyncio.Semaphore(4)
    case_limiter = asyncio.Semaphore(1)
    results = []
    predictions = []
    status = RunStatus.FAILED
    try:
        for case in prepared:
            cid = case.case_id
            if condition == "no-task-contract":
                request, lineage = _ablation_request(case.input)
                folder.write_immutable(f"inputs/lineage/{cid}.json", canonical_json(lineage))
                result = await task_contract_ablation.run_case(
                    output=folder.root, case_id=cid, request=request, runtime=runtime, limiter=model_limiter,
                    skill=skills["methods"],
                    timeout_seconds=timeout_seconds,
                )
            else:
                kwargs = dict(runtime=runtime, folder=folder, stages=stages,
                              model_limiter=model_limiter, case_limiter=case_limiter)
                if condition == "full":
                    terminal = await live._run_staged_candidate(
                        case, **kwargs, domain_skill_texts=skills,
                        review_skill_text=skills["review"], overall_skill_text=skills["overall"],
                    )
                else:
                    terminal = await live._run_candidate(case, **kwargs)
                prediction = terminal.prediction
                domains = prediction.domains if prediction is not None else terminal.readable_domains
                predictions.append({
                    "case_id": cid, "domains": {item.domain: item.judgment for item in domains},
                    "overall_downgrade_levels": prediction.overall_downgrade_levels if prediction else None,
                })
                result = {
                    "case_id": cid,
                    "status": "COMPLETE" if prediction and not terminal.evidence_issues else
                    "COMPLETED_WITH_ISSUES" if domains else "FAILED",
                    "failure_code": terminal.failure_code,
                    "failure_detail": terminal.failure_detail,
                    "model_call_count": terminal.model_call_count,
                    "prediction": prediction.model_dump(mode="json") if prediction else None,
                    "readable_domains": [item.model_dump(mode="json") for item in domains],
                    "evidence_issues": [issue.model_dump(mode="json") for issue in terminal.evidence_issues],
                }
            artifact = folder.write_immutable(f"results/{cid}.json", canonical_json(result))
            results.append({"case_id": cid, "status": result["status"], "path": artifact.path, "sha256": artifact.sha256})
        if all(row["status"] == "COMPLETE" for row in results):
            status = RunStatus.SUCCEEDED
        elif any(row["status"] not in {"FAILED", "NO_READABLE_ANSWER"} for row in results):
            status = RunStatus.COMPLETED_WITH_ISSUES
    finally:
        if condition != "no-task-contract":
            folder.write_immutable("predictions.json", canonical_json(predictions))
        execution = folder.write_immutable("execution.json", canonical_json({
            "benchmark": BENCHMARK, "run_id": run_id, "condition": condition, "status": status.value,
            "selected_case_count": len(prepared), "results": results,
        }))
        manifest.set_run_status(run_id, status, finished_at=utc_now(),
                                final_output_path=execution.path, final_output_sha256=execution.sha256)


def load_references() -> list[dict]:
    bundle = json.loads((RESEARCH_ROOT / "references/ec-ex80.json").read_bytes())
    cases = bundle["cases"]
    valid = set(get_args(DomainJudgment))
    if (
        bundle["benchmark"] != BENCHMARK
        or bundle["reference_kind"] != "SEALED_AI_ADJUDICATION"
        or tuple(row["case_id"] for row in cases) != case_ids()
        or any(set(row["domains"]) != set(GRADE_DOMAINS) or set(row["domains"].values()) - valid
               or type(row["overall_downgrade_levels"]) is not int
               or not 0 <= row["overall_downgrade_levels"] <= 3 for row in cases)
    ):
        raise ValueError("EC-ex80 references differ from the frozen five-domain case inventory")
    return cases


def score(rows: list[dict]) -> dict:
    reference = load_references()
    predictions = {row["case_id"]: row for row in rows}
    if len(predictions) != len(rows) or predictions.keys() - set(case_ids()):
        raise ValueError("Predictions contain duplicate or unknown EC-ex80 case IDs")
    domain_exact = downgrade_exact = case_exact = 0
    errors = []
    for case in reference:
        prediction = predictions.get(case["case_id"], {})
        domains = prediction.get("domains", {})
        matches = sum(domains.get(domain) == judgment for domain, judgment in case["domains"].items())
        overall = prediction.get("overall_downgrade_levels")
        valid = type(overall) is int and 0 <= overall <= 3
        exact = valid and overall == case["overall_downgrade_levels"]
        domain_exact += matches
        downgrade_exact += exact
        case_exact += matches == len(GRADE_DOMAINS) and exact
        if valid:
            errors.append(abs(overall - case["overall_downgrade_levels"]))
    return {
        "benchmark": BENCHMARK,
        "reference_kind": "SEALED_AI_ADJUDICATION",
        "domain_matches": domain_exact,
        "domain_denominator": len(reference) * len(GRADE_DOMAINS),
        "domain_agreement": domain_exact / (len(reference) * len(GRADE_DOMAINS)),
        "downgrade_matches": downgrade_exact,
        "case_denominator": len(reference),
        "downgrade_agreement": downgrade_exact / len(reference),
        "case_matches": case_exact,
        "case_exact": case_exact / len(reference),
        "downgrade_mae": sum(errors) / len(errors) if errors else None,
        "available_downgrades": len(errors),
    }
