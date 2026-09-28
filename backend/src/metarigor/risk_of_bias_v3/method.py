from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from .models import AlgorithmResult, Domain, Judgment, NormalizedAnswer, canonical_json
_SPEC_PATH = Path(__file__).with_name("method_spec.json")


_SPEC_BYTES = _SPEC_PATH.read_bytes()


_IMPLEMENTATION_BYTES = Path(__file__).read_bytes()


METHOD_SPEC_SHA256 = hashlib.sha256(_SPEC_BYTES).hexdigest()


METHOD_IMPLEMENTATION_SHA256 = hashlib.sha256(_IMPLEMENTATION_BYTES).hexdigest()


METHOD_FINGERPRINT = hashlib.sha256(
    canonical_json(
        {
            "method_spec_sha256": METHOD_SPEC_SHA256,
            "deterministic_implementation_sha256": METHOD_IMPLEMENTATION_SHA256,
        }
    )
).hexdigest()


METHOD_SPEC: dict[str, Any] = json.loads(_SPEC_BYTES)


_POS = {"YES", "PROBABLY_YES"}


_NEG = {"NO", "PROBABLY_NO"}


_NI = "NO_INFORMATION"


_NA = "NOT_APPLICABLE"


_ALL = _POS | _NEG | {_NI}


_QUESTION_ALLOWED = {
    question: set(
        METHOD_SPEC["question_specific_allowed_answers"].get(
            question, METHOD_SPEC["allowed_answers"]
        )
    )
    for domain in ("D1", "D2", "D3", "D4", "D5")
    for question in METHOD_SPEC["domains"][domain]["questions"]
}


class MethodInputError(ValueError):
    """Input does not conform to the fixed RoB 2 method specification."""


def question_ids(domain: Domain) -> tuple[str, ...]:
    return tuple(METHOD_SPEC["domains"][domain]["questions"])


def method_input_answers(domain: Domain, raw_answers: Mapping[str, str]) -> dict[str, str]:
    """Map model vocabulary to the input vocabulary allowed by the fixed method."""
    answers = dict(raw_answers)
    if domain == "D3" and answers.get("3.2") == "NO_INFORMATION":
        answers["3.2"] = "PROBABLY_NO"
    return answers


def normalize_applicability(
    domain: Domain, answers: Mapping[str, str]
) -> dict[str, NormalizedAnswer]:
    expected = set(question_ids(domain))
    if set(answers) != expected:
        raise MethodInputError(
            f"question closure mismatch: expected={sorted(expected)} actual={sorted(answers)}"
        )
    if any(value not in _ALL for value in answers.values()):
        raise MethodInputError("model answers must use the fixed answer vocabulary")
    normalized: dict[str, NormalizedAnswer] = dict(answers)  # type: ignore[assignment]

    def na(*ids: str) -> None:
        for item in ids:
            normalized[item] = _NA

    if domain == "D2":
        if normalized["2.1"] in _NEG and normalized["2.2"] in _NEG:
            na("2.3", "2.4", "2.5")
        elif normalized["2.3"] not in _POS:
            na("2.4", "2.5")
        elif normalized["2.4"] not in (_POS | {_NI}):
            na("2.5")
        if normalized["2.6"] in _POS:
            na("2.7")
    elif domain == "D3":
        if normalized["3.1"] in _POS:
            na("3.2", "3.3", "3.4")
        elif normalized["3.2"] in _POS:
            na("3.3", "3.4")
        elif normalized["3.3"] not in (_POS | {_NI}):
            na("3.4")
    elif domain == "D4":
        first_two_clear = normalized["4.1"] in (_NEG | {_NI}) and normalized["4.2"] in (
            _NEG | {_NI}
        )
        if not first_two_clear:
            na("4.3", "4.4", "4.5")
        elif normalized["4.3"] not in (_POS | {_NI}):
            na("4.4", "4.5")
        elif normalized["4.4"] not in (_POS | {_NI}):
            na("4.5")
    elif domain not in {"D1", "D5"}:
        raise MethodInputError(f"unsupported domain: {domain}")
    invalid = {
        question: answer
        for question, answer in normalized.items()
        if answer != _NA and answer not in _QUESTION_ALLOWED[question]
    }
    if invalid:
        raise MethodInputError(f"question-specific answer vocabulary mismatch: {invalid}")
    return normalized


def _result(
    domain: Domain,
    answers: Mapping[str, NormalizedAnswer],
    visited: list[str],
    edges: list[str],
    judgment: Judgment,
) -> AlgorithmResult:
    return AlgorithmResult(
        method_fingerprint=METHOD_FINGERPRINT,
        domain=domain,
        input_answers_sha256=hashlib.sha256(canonical_json(dict(answers))).hexdigest(),
        visited_nodes=tuple(visited),
        selected_edges=tuple(edges),
        proposed_judgment=judgment,
    )


def compute_domain(domain: Domain, raw_answers: Mapping[str, str]) -> AlgorithmResult:
    a = normalize_applicability(domain, raw_answers)
    visited = [f"{domain}:START"]
    edges: list[str] = []

    def take(node: str, condition: str) -> None:
        visited.append(node)
        edges.append(f"{node}:{condition}")

    if domain == "D1":
        take("D1:Q1.2", a["1.2"])
        if a["1.2"] in _NEG:
            return _result(domain, a, visited, edges, "HIGH")
        if a["1.2"] == _NI:
            take("D1:Q1.3_AFTER_NI", a["1.3"])
            judgment = "HIGH" if a["1.3"] in _POS else "SOME_CONCERNS"
            return _result(domain, a, visited, edges, judgment)
        take("D1:Q1.1", a["1.1"])
        if a["1.1"] in _NEG:
            return _result(domain, a, visited, edges, "SOME_CONCERNS")
        take("D1:Q1.3", a["1.3"])
        return _result(domain, a, visited, edges, "SOME_CONCERNS" if a["1.3"] in _POS else "LOW")

    if domain == "D2":
        take("D2:PART1_AWARENESS", f"{a['2.1']}|{a['2.2']}")
        if a["2.3"] == _NA or a["2.3"] in _NEG:
            part1: Judgment = "LOW"
        elif a["2.3"] == _NI:
            part1 = "SOME_CONCERNS"
        else:
            take("D2:PART1_AFFECT_OUTCOME", a["2.4"])
            if a["2.4"] in _NEG:
                part1 = "SOME_CONCERNS"
            elif a["2.5"] in _POS:
                take("D2:PART1_BALANCED", a["2.5"])
                part1 = "SOME_CONCERNS"
            else:
                take("D2:PART1_BALANCED", a["2.5"])
                part1 = "HIGH"
        take("D2:PART2_ANALYSIS", a["2.6"])
        if a["2.6"] in _POS:
            part2: Judgment = "LOW"
        elif a["2.7"] in _NEG:
            take("D2:PART2_IMPACT", a["2.7"])
            part2 = "SOME_CONCERNS"
        else:
            take("D2:PART2_IMPACT", a["2.7"])
            part2 = "HIGH"
        severity = {"LOW": 0, "SOME_CONCERNS": 1, "HIGH": 2}
        return _result(domain, a, visited, edges, max((part1, part2), key=severity.__getitem__))

    if domain == "D3":
        for question in ("3.1", "3.2", "3.3", "3.4"):
            if a[question] != _NA:
                take(f"D3:Q{question}", a[question])
        if a["3.1"] in _POS or a["3.2"] in _POS or a["3.3"] in _NEG:
            judgment = "LOW"
        elif a["3.4"] in _NEG:
            judgment = "SOME_CONCERNS"
        else:
            judgment = "HIGH"
        return _result(domain, a, visited, edges, judgment)

    if domain == "D4":
        for question in ("4.1", "4.2", "4.3", "4.4", "4.5"):
            if a[question] != _NA:
                take(f"D4:Q{question}", a[question])
        if a["4.1"] in _POS or a["4.2"] in _POS or a["4.5"] in (_POS | {_NI}):
            judgment = "HIGH"
        elif a["4.1"] == _NI or a["4.2"] == _NI or a["4.4"] in (_POS | {_NI}):
            judgment = "SOME_CONCERNS"
        else:
            judgment = "LOW"
        return _result(domain, a, visited, edges, judgment)

    for question in ("5.1", "5.2", "5.3"):
        take(f"D5:Q{question}", a[question])
    if a["5.2"] in _POS or a["5.3"] in _POS:
        judgment = "HIGH"
    elif a["5.1"] in _POS and a["5.2"] in _NEG and a["5.3"] in _NEG:
        judgment = "LOW"
    else:
        judgment = "SOME_CONCERNS"
    return _result(domain, a, visited, edges, judgment)


def compute_overall(domain_judgments: Sequence[Judgment]) -> AlgorithmResult:
    if len(domain_judgments) != 5:
        raise MethodInputError("overall judgment requires exactly five domain judgments")
    if "HIGH" in domain_judgments:
        judgment: Judgment = "HIGH"
        edge = "OVERALL:any-high"
    elif all(item == "LOW" for item in domain_judgments):
        judgment = "LOW"
        edge = "OVERALL:all-low"
    else:
        judgment = "SOME_CONCERNS"
        edge = "OVERALL:some-and-no-high"
    payload = list(domain_judgments)
    return AlgorithmResult(
        method_fingerprint=METHOD_FINGERPRINT,
        domain="OVERALL",
        input_answers_sha256=hashlib.sha256(canonical_json(payload)).hexdigest(),
        visited_nodes=("OVERALL:START", "OVERALL:DOMAIN_JUDGMENTS"),
        selected_edges=(edge,),
        proposed_judgment=judgment,
    )

