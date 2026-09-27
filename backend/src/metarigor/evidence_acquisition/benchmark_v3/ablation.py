from __future__ import annotations

"""五案例 EA 封存结果的离线消融与主报告材料实验输入；不重写原 Run。"""


import hashlib
import json
from metarigor.local_run import RunFolder, canonical_json
def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def valid_candidate(candidate):
    return bool(candidate and candidate.get("grounding_status") == "RESOLVED")


def reconcile(fast, deep, exclusion_ids, adjudicator=None, *, adjudicate=True):
    """复现 V3 的决定规则；无裁断条件仅把真实分歧留为 UNCERTAIN。"""
    if not valid_candidate(fast) or not valid_candidate(deep):
        return "UNASSESSED", "INVOCATION_FAILURE"
    common = set(fast["criterion_ids"]) & set(deep["criterion_ids"]) & set(exclusion_ids)
    if fast["decision"] == deep["decision"] and (fast["decision"] != "EXCLUDE" or common):
        return fast["decision"], (
            "ASSESSOR_UNCERTAIN" if fast["decision"] == "UNCERTAIN" else "ASSESSOR_AGREEMENT"
        )
    if not adjudicate:
        return "UNCERTAIN", "DISAGREEMENT_WITHOUT_ADJUDICATION"
    if valid_candidate(adjudicator):
        return adjudicator["decision"], "FIXED_ADJUDICATOR"
    return "UNASSESSED", "INVOCATION_FAILURE"


def ground_output(output, payload, folder: RunFolder):
    """沿用文档局部 segment 定位，逐字回取 representation 后绑定 Source Span。"""
    semantic = payload["semantic_input"]
    criteria = {c["criterion_id"]: c for c in semantic["protocol"]["criteria"]}
    ids = output["criterion_ids"]
    if any(i not in criteria for i in ids):
        raise ValueError("unknown Protocol criterion")
    if output["decision"] == "EXCLUDE" and (
        not ids or any(criteria[i]["disposition"] != "EXCLUDE" for i in ids)
    ):
        raise ValueError("EXCLUDE requires only formal exclusion criteria")
    docs = {d["document_id"]: d for d in semantic["packet_manifest"]["documents"]}
    spans = []
    for ref in output["supporting_source_refs"]:
        doc = docs.get(ref["document_id"])
        if doc is None:
            raise ValueError("evidence outside selected primary documents")
        view = json.loads(
            folder.read_verified(
                "baseline/" + doc["selected_view_path"], doc["selected_view_sha256"]
            )
        )
        matches = [s for s in view["segments"] if s["segment_id"] == ref["segment_id"]]
        if len(matches) != 1:
            raise ValueError("unknown or duplicate document-local segment")
        seg = matches[0]
        text = folder.read_verified(
            "baseline/" + doc["representation_path"], doc["representation_sha256"]
        ).decode()
        if text[seg["start_offset"] : seg["end_offset"]] != seg["text"]:
            raise ValueError("source quote/offset mismatch")
        span = {
            "document_id": doc["document_id"],
            "segment_id": ref["segment_id"],
            "representation_path": "baseline/" + doc["representation_path"],
            "representation_sha256": doc["representation_sha256"],
            "start_offset": seg["start_offset"],
            "end_offset": seg["end_offset"],
            "quote": seg["text"],
            "quote_sha256": hashlib.sha256(seg["text"].encode()).hexdigest(),
        }
        span["span_id"] = "span:" + digest(span)[:24]
        if span not in spans:
            spans.append(span)
    if not spans:
        raise ValueError("no grounded evidence")
    return spans


