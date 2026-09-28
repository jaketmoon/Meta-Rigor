from __future__ import annotations

"""Two bounded RoB ablations: freeze historical inputs, generate two candidate sets, and score only new artifacts."""


import asyncio
import copy
import hashlib
import inspect
import json
from collections import Counter, defaultdict
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from jsonschema import Draft202012Validator
from metarigor.adapters.agent_runtime.schema_agent import ClaudeSchemaAgent, DirectSchemaAgent
from metarigor.local_run import RunFolder, RunManifest, RunStatus, canonical_json
from metarigor.local_run.runner import StageIssue, StageOutcome, StageRunner, StageStatus, utc_now
from metarigor.risk_of_bias_v3 import method
from metarigor.risk_of_bias_v3.evidence import QUESTION_PROBES, build_absence_search_record, resolve_quote
from metarigor.risk_of_bias_v3.models import EvidenceQuoteCandidate, StudySourceBundle
from metarigor.schema_agent import SchemaAgentOutputError, SchemaAgentRequest, SchemaAgentResult
ARMS = ("minus_decomposition", "minus_programmatic_decision")


DOMAINS = ("D1", "D2", "D3", "D4", "D5")


ANSWERS = ("YES", "PROBABLY_YES", "PROBABLY_NO", "NO", "NO_INFORMATION")


JUDGMENTS = ("LOW", "SOME_CONCERNS", "HIGH")


MISSING = "MISSING"


class MeasuredDirectAgent(DirectSchemaAgent):
    """Read response telemetry only; the request body matches the historical direct runtime."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics_context = ContextVar("rob_response_metrics", default=None)
        self._client.event_hooks["response"].append(self.capture_metrics)

    async def capture_metrics(self, response):
        await response.aread()
        metrics = self.metrics_context.get()
        if metrics is None:
            return
        metrics["http_status"] = response.status_code
        try:
            body = response.json()
            metrics["response_model"] = body.get("model")
            metrics["usage"] = body.get("usage")
            metrics["finish_reason"] = body.get("choices", [{}])[0].get("finish_reason")
        except (ValueError, AttributeError, IndexError, TypeError, KeyError):
            pass

    async def invoke(self, request):
        metrics = {}
        token = self.metrics_context.set(metrics)
        try:
            result = await super().invoke(request)
            return SchemaAgentResult(
                payload=result.payload,
                raw_model_output=result.raw_model_output,
                metrics=metrics,
            )
        except Exception as error:
            error.metrics = metrics
            raise
        finally:
            self.metrics_context.reset(token)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json(value))


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def inline_schema(schema):
    definitions = schema.get("$defs", {})

    def visit(value):
        if isinstance(value, list):
            return [visit(v) for v in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            return visit(definitions[value["$ref"].split("/")[-1]])
        return {k: visit(v) for k, v in value.items() if k not in {"$defs", "default"}}

    return visit(schema)


def decision_rules():
    # Pin the original method source as verifiable rule documentation; expose no tools or Python execution to the model.
    return (
        "Read this fixed method as rules to execute yourself. POS={YES,PROBABLY_YES}; "
        "NEG={NO,PROBABLY_NO}; NI=NO_INFORMATION; NA=NOT_APPLICABLE. "
        "_result(..., judgment) returns that judgment; take() only records provenance. "
        "First apply method_input_answers, then normalize_applicability for final_answer, "
        "then compute_domain to choose domain_judgment. No external tools.\n"
        + "_POS={'YES','PROBABLY_YES'}\n_NEG={'NO','PROBABLY_NO'}\n"
        + "_NI='NO_INFORMATION'\n_NA='NOT_APPLICABLE'\n_ALL=_POS|_NEG|{_NI}\n"
        + "question_ids(domain) uses this exact inventory:\n"
        + json.dumps({d: list(method.question_ids(d)) for d in DOMAINS})
        + "\n_QUESTION_ALLOWED="
        + json.dumps({q: sorted(v) for q, v in method._QUESTION_ALLOWED.items()})
        + "\n"
        + inspect.getsource(method.method_input_answers)
        + inspect.getsource(method.normalize_applicability)
        + inspect.getsource(method.compute_domain)
    )


def decision_prompt(original):
    prompt = (
        original.replace(
            "do not emit NOT_APPLICABLE, a domain judgment, or an overall judgment.",
            "Keep raw answer in the original vocabulary; also emit final_answer and domain_judgment.",
        )
        .replace(
            "The program alone computes conditional applicability and ignores\n"
            "the raw answer for a question it determines is not applicable.",
            "You alone compute conditional applicability and place NOT_APPLICABLE in final_answer "
            "for a skipped question. The program will not correct final_answer or domain_judgment.",
        )
        .replace(
            "has exactly question_id, answer, evidence and absence_probe_id;",
            "has exactly question_id, answer, final_answer, evidence and absence_probe_id;",
        )
    )
    return prompt + "\nReturn answers and domain_judgment.\n" + decision_rules()


def output_schema(item, decision=False):
    schema = inline_schema(item["payload"]["output_schema"])
    if decision:
        answer = schema["properties"]["answers"]["items"]
        answer["properties"]["final_answer"] = {"enum": [*ANSWERS, "NOT_APPLICABLE"]}
        answer["required"].append("final_answer")
        schema["properties"]["domain_judgment"] = {"enum": list(JUDGMENTS)}
        schema["required"].append("domain_judgment")
    Draft202012Validator.check_schema(schema)
    return schema


def make_jobs(case, arm):
    if arm == "full":
        return [{"key": i["key"], "item_keys": [i["key"]],
                 "prompt": i["payload"]["prompt"], "input": i["payload"]["semantic_input"],
                 "schema": i["payload"]["output_schema"], "skill_text": i["payload"]["skill_text"]}
                for i in case["items"]]
    if arm == ARMS[1]:
        return [
            {
                "key": i["key"],
                "item_keys": [i["key"]],
                "prompt": decision_prompt(i["payload"]["prompt"]),
                "input": i["payload"]["semantic_input"],
                "schema": output_schema(i, True),
                "skill_text": i["payload"]["skill_text"],
            }
            for i in case["items"]
        ]
    by_study = defaultdict(list)
    for i in case["items"]:
        by_study[i["study_key"]].append(i)
    jobs = []
    for study, items in sorted(by_study.items()):
        documents, semantic_items = {}, {}
        for i in items:
            sem = copy.deepcopy(i["payload"]["semantic_input"])
            for doc in sem.pop("documents"):
                if (
                    doc["document_key"] in documents
                    and documents[doc["document_key"]] != doc
                ):
                    raise ValueError("Study documents differ across domain inputs")
                documents[doc["document_key"]] = doc
            semantic_items[i["key"]] = sem
        prompts = list(dict.fromkeys(i["payload"]["prompt"] for i in items))
        prompt = (
            "Assess ONE Study jointly in ONE response. Each keyed item below is an output slot, "
            "not a separate call. Read the complete shared documents once and answer every slot. "
            "Keep each slot's exact scope; do not transfer target-specific conclusions. "
            "Return {items: {slot_key: {answers: [...]}}}. No domain or overall judgments.\n"
            + "\n".join(
                p.replace(
                    "one bounded RoB 2 domain item", "the bounded Study domain slots"
                )
                for p in prompts
            )
        )
        schema = obj({"items": obj({i["key"]: output_schema(i) for i in items})})
        jobs.append(
            {
                "key": study,
                "item_keys": [i["key"] for i in items],
                "prompt": prompt,
                "input": {
                    "study_key": study,
                    "documents": list(documents.values()),
                    "items": semantic_items,
                },
                "schema": schema,
                "skill_text": items[0]["payload"]["skill_text"],
            }
        )
    return jobs


def closed_input_schema(value):
    # Exact inputs are the only allowed values for this Stage; explicitly close nested objects as well.
    if isinstance(value, dict):
        return obj({k: closed_input_schema(v) for k, v in value.items()})
    if isinstance(value, list):
        return {"type": "array", "const": value}
    return {"const": value}


def retain_fields(item, candidate, decision):
    """Keep only schema-valid fields; do not change answers or use first/last selection for duplicate questions."""
    result = {"raw_answers": {}, "answers": {}, "evidence": {}, "issues": []}
    schema = output_schema(item, decision)
    record_schema = schema["properties"]["answers"]["items"]
    records = candidate.get("answers", []) if isinstance(candidate, dict) else []
    if not isinstance(records, list):
        records = []
    counts = Counter(
        r.get("question_id")
        for r in records
        if isinstance(r, dict) and isinstance(r.get("question_id"), str)
    )
    for q in method.question_ids(item["payload"]["semantic_input"]["domain"]):
        matching = [
            r for r in records if isinstance(r, dict) and r.get("question_id") == q
        ]
        if counts[q] != 1:
            result["issues"].append(f"QUESTION_MISSING_OR_DUPLICATE:{q}")
            continue
        r = matching[0]
        for field, output_key in (
            ("answer", "raw_answers"),
            ("final_answer", "answers"),
        ):
            if field == "final_answer" and not decision:
                continue
            if Draft202012Validator(record_schema["properties"][field]).is_valid(
                r.get(field)
            ):
                result[output_key][q] = r[field]
            else:
                result["issues"].append(f"INVALID_FIELD:{q}:{field}")
        if all(
            Draft202012Validator(record_schema["properties"][f]).is_valid(r.get(f))
            for f in ("evidence", "absence_probe_id")
        ):
            result["evidence"][q] = {
                "evidence": r["evidence"],
                "absence_probe_id": r["absence_probe_id"],
            }
        else:
            result["issues"].append(f"INVALID_EVIDENCE:{q}")
    domain = item["payload"]["semantic_input"]["domain"]
    try:
        method_answers = method.method_input_answers(domain, result["raw_answers"])
        normalized = method.normalize_applicability(domain, method_answers)
        algorithm = method.compute_domain(domain, method_answers).proposed_judgment
    except ValueError:
        normalized, algorithm = {}, MISSING
    result["diagnostic_program_answers"] = normalized
    result["diagnostic_program_judgment"] = algorithm
    if decision:
        label = (
            candidate.get("domain_judgment") if isinstance(candidate, dict) else None
        )
        result["judgment"] = label if label in JUDGMENTS else MISSING
        if result["judgment"] == MISSING:
            result["issues"].append("MISSING_DOMAIN_JUDGMENT")
        for q, value in normalized.items():
            if q in result["answers"] and result["answers"][q] != value:
                result["issues"].append(f"MODEL_APPLICABILITY_MISMATCH:{q}")
        if algorithm != MISSING and result["judgment"] not in {MISSING, algorithm}:
            result["issues"].append("MODEL_DOMAIN_RULE_MISMATCH")
    else:
        result["answers"] = normalized or dict(result["raw_answers"])
        result["judgment"] = algorithm
    return result


def evidence_audit(item, record, bundle_data):
    bundle = StudySourceBundle.model_validate(bundle_data)
    docs = {d.document_key: d for d in bundle.members}
    audits = {}
    for q, answer in record["answers"].items():
        if answer == "NOT_APPLICABLE":
            audits[q] = {"status": "NOT_APPLICABLE", "spans": [], "absence": None}
            continue
        evidence = record["evidence"].get(q, {})
        spans = []
        for quote in evidence.get("evidence", []):
            candidate = EvidenceQuoteCandidate(**quote, purpose=q)
            spans.append(
                resolve_quote(candidate, docs[candidate.document_key]).model_dump(
                    mode="json"
                )
            )
        probe = evidence.get("absence_probe_id")
        absence = None
        valid_mode = not (answer == "NO_INFORMATION" and (spans or probe is None))
        if probe is not None:
            valid_mode &= (
                probe in QUESTION_PROBES[q]
                and answer in {"NO", "PROBABLY_NO", "NO_INFORMATION"}
                and not spans
            )
            if valid_mode:
                absence = build_absence_search_record(
                    probe_id=probe,
                    study_key=bundle.study_key,
                    domain=item["payload"]["semantic_input"]["domain"],
                    bundle=bundle,
                    required_roles_closed=not item["payload"]["semantic_input"][
                        "missing_document_roles"
                    ]
                    and not bundle.missing_document_keys,
                ).model_dump(mode="json")
        supported = valid_mode and (
            (
                probe is None
                and answer != "NO_INFORMATION"
                and any(s["match_kind"] == "EXACT" for s in spans)
            )
            or (absence is not None and absence["conclusion"] == "NO_HIT")
        )
        audits[q] = {
            "status": "SUPPORTED" if supported else "UNSUPPORTED",
            "spans": spans,
            "absence": absence,
        }
    return audits


async def invoke_job(runner, folder, job, runtime, limiter):
    payload = {
        "job": job,
        "runtime": runtime.binding("worker"),
        "method_fingerprint": method.METHOD_FINGERPRINT,
    }

    async def execute(_):
        Draft202012Validator(closed_input_schema(job["input"])).validate(job["input"])
        async with limiter:
            started = perf_counter()
            try:
                response = await runtime.invoke(
                    SchemaAgentRequest(
                        template_id="rob-two-ablations",
                        model_role="worker",
                        prompt=job["prompt"],
                        input_payload=job["input"],
                        output_schema=job["schema"],
                        skill_text=job["skill_text"],
                    )
                )
            except SchemaAgentOutputError as error:
                # Read fields from the same response; this is not a model retry or repair.
                raw = error.raw_model_output
                candidate = None
                if isinstance(raw, str):
                    try:
                        candidate = ClaudeSchemaAgent._decode_json_text(raw)
                    except ValueError:
                        pass
                return StageOutcome(
                    payload={
                        "candidate": candidate,
                        "error": str(error),
                        "metrics": getattr(error, "metrics", None),
                        "elapsed_seconds": perf_counter() - started,
                    },
                    raw_model_output=raw,
                    status=StageStatus.COMPLETED_WITH_ISSUES
                    if isinstance(candidate, dict)
                    else StageStatus.FAILED,
                    issues=(StageIssue("MODEL_OUTPUT_FAILURE", "ERROR", str(error)),),
                )
            return StageOutcome(
                payload={
                    "candidate": response.payload,
                    "elapsed_seconds": perf_counter() - started,
                    "metrics": response.metrics,
                },
                raw_model_output=response.raw_model_output,
            )

    stage = await runner.execute(
        stage="model_call", item_id=job["key"], payload=payload, executor=execute
    )
    data = json.loads(folder.read_verified(stage.output_path, stage.output_sha256))
    print(
        json.dumps({"run": runner.run_id, "item": job["key"], "status": stage.status}),
        flush=True,
    )
    return data.get("candidate"), asdict(stage)


def copy_inputs(case, folder):
    # Frozen study source bundles contain the full text and verified source identity.
    saved = folder.write_immutable("inputs/case.json", canonical_json(case))
    return [asdict(saved)]


async def run_case(case, arm, root, runtime, limiter):
    start = perf_counter()
    folder = RunFolder.create(root / arm / case["case_id"])
    run_id = str(uuid4())
    manifest = RunManifest(folder.manifest_path)
    manifest.initialize()
    jobs = make_jobs(case, arm)
    snapshot = {
        "case_id": case["case_id"],
        "arm": arm,
        "targets": case["targets"],
        "cells": case["cells"],
        "source_bindings": copy_inputs(case, folder),
        "jobs": jobs,
        "runtime": runtime.binding("worker"),
        "method_fingerprint": method.METHOD_FINGERPRINT,
        "code_sha256": digest(Path(__file__).read_bytes()),
    }
    saved = folder.write_immutable("request.json", canonical_json(snapshot))
    manifest.create_run(
        run_id=run_id,
        specialist="risk-of-bias-ablation",
        input_path=saved.path,
        input_sha256=saved.sha256,
        started_at=utc_now(),
    )
    manifest.set_run_status(run_id, RunStatus.RUNNING)
    runner = StageRunner(run_id=run_id, folder=folder, manifest=manifest)
    outputs = await asyncio.gather(
        *(invoke_job(runner, folder, j, runtime, limiter) for j in jobs)
    )
    items, stages = {}, []
    indexed = {i["key"]: i for i in case["items"]}
    for job, (candidate, stage) in zip(jobs, outputs, strict=True):
        stages.append(stage)
        values = (
            candidate.get("items", {})
            if isinstance(candidate, dict) and arm == ARMS[0]
            else {}
        )
        for key in job["item_keys"]:
            value = (
                values.get(key)
                if isinstance(values, dict) and arm == ARMS[0]
                else candidate
            )
            record = retain_fields(indexed[key], value, arm == ARMS[1])
            record["model_stage"] = stage
            record["evidence_audit"] = evidence_audit(
                indexed[key], record, case["bundles"][indexed[key]["study_key"]]
            )
            items[key] = record
    targets = []
    for target in case["targets"]:
        labels = {d: items[k]["judgment"] for d, k in target["items"].items()}
        overall = MISSING
        if MISSING not in labels.values():
            if arm != ARMS[1]:
                overall = method.compute_overall(
                    list(labels.values())
                ).proposed_judgment
            else:
                job = {
                    "key": f"overall:{target['study_key']}:{target['target_id']}",
                    "item_keys": [],
                    "input": {"domains": labels},
                    "skill_text": None,
                    "prompt": "Apply this fixed rule yourself to the five supplied Domain labels: "
                    "any HIGH -> HIGH; all LOW -> LOW; otherwise SOME_CONCERNS. Return overall_judgment only.",
                    "schema": obj({"overall_judgment": {"enum": list(JUDGMENTS)}}),
                }
                result, stage = await invoke_job(runner, folder, job, runtime, limiter)
                stages.append(stage)
                if (
                    isinstance(result, dict)
                    and result.get("overall_judgment") in JUDGMENTS
                ):
                    overall = result["overall_judgment"]
        targets.append(
            target
            | {
                "domain_labels": labels,
                "overall": overall,
                "overall_rule_mismatch": MISSING not in labels.values()
                and overall != MISSING
                and overall
                != method.compute_overall(list(labels.values())).proposed_judgment,
            }
        )
    telemetry = []
    for stage in stages:
        value = json.loads(
            folder.read_verified(stage["output_path"], stage["output_sha256"])
        )
        metrics = value.get("metrics")
        if metrics is None and isinstance(value.get("error"), dict):
            metrics = value["error"].get("metrics")
        telemetry.append(metrics)
    usages = [m.get("usage") for m in telemetry if isinstance(m, dict)]
    usage_complete = len(usages) == len(stages) and all(
        isinstance(u, dict)
        and all(
            isinstance(u.get(k), int) for k in ("prompt_tokens", "completion_tokens")
        )
        for u in usages
    )
    package = {
        "schema_version": "rob-two-ablations-v1",
        "case_id": case["case_id"],
        "arm": arm,
        "run_id": run_id,
        "items": items,
        "targets": targets,
        "cells": case["cells"],
        "stages": stages,
        "model_calls": len(stages),
        "elapsed_seconds": perf_counter() - start,
        "input_tokens": sum(u["prompt_tokens"] for u in usages)
        if usage_complete
        else None,
        "output_tokens": sum(u["completion_tokens"] for u in usages)
        if usage_complete
        else None,
        "cost_usd": None,
        "usage_status": "OBSERVED" if usage_complete else "PARTIAL_OR_UNAVAILABLE",
        "response_telemetry": telemetry,
    }

    async def materialize(_):
        issues = [
            StageIssue("CANDIDATE_ISSUE", "ERROR", f"{key}:{issue}")
            for key, record in items.items()
            for issue in record["issues"]
        ]
        issues += [
            StageIssue("UNSUPPORTED_EVIDENCE", "ERROR", f"{key}:{q}")
            for key, record in items.items()
            for q, audit in record["evidence_audit"].items()
            if audit["status"] == "UNSUPPORTED"
        ]
        return StageOutcome(
            payload=package,
            status=StageStatus.COMPLETED_WITH_ISSUES
            if issues
            else StageStatus.SUCCEEDED,
            issues=tuple(issues),
        )

    final = await runner.execute(
        stage="seal_candidate",
        item_id="case",
        payload={"stages": stages},
        executor=materialize,
    )
    readable = any(r["answers"] or r["judgment"] != MISSING for r in items.values())
    manifest.set_run_status(
        run_id,
        RunStatus.COMPLETED_WITH_ISSUES if readable else RunStatus.FAILED,
        finished_at=utc_now(),
        final_output_path=final.output_path,
        final_output_sha256=final.output_sha256,
    )
    save(folder.root / "candidate.json", package)
    return {
        "case_id": case["case_id"],
        "arm": arm,
        "run_id": run_id,
        "path": str(folder.root.relative_to(root)),
        "candidate_sha256": digest(canonical_json(package)),
    }

