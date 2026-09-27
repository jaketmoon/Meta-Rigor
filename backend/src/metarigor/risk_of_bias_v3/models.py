from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


Key = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,239}$")]


Domain = Literal["D1", "D2", "D3", "D4", "D5"]


NormalizedAnswer = Literal[
    "YES", "PROBABLY_YES", "PROBABLY_NO", "NO", "NO_INFORMATION", "NOT_APPLICABLE"
]


Judgment = Literal["LOW", "SOME_CONCERNS", "HIGH"]


def canonical_json(value: BaseModel | dict[str, Any] | list[Any]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class RoBModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self)).hexdigest()


class ExtractorConfigEntry(RoBModel):
    name: str = Field(min_length=1, max_length=200)
    value: str = Field(max_length=2_000)


class PreparedDocument(RoBModel):
    document_key: Key
    report_key: Key
    document_role: Literal[
        "RESULT_REPORT",
        "SUPPLEMENT",
        "PROTOCOL",
        "REGISTRY",
        "SAP",
        "COMPANION",
        "CORRECTION",
        "OTHER",
    ]
    source_path: str = Field(min_length=1, max_length=10_000)
    source_sha256: Sha256
    representation_path: str = Field(min_length=1, max_length=10_000)
    representation_sha256: Sha256
    extractor_name: str = Field(min_length=1, max_length=200)
    extractor_version: str = Field(min_length=1, max_length=200)
    extractor_config: tuple[ExtractorConfigEntry, ...] = ()
    full_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def representation_is_bound(self) -> PreparedDocument:
        if hashlib.sha256(self.full_text.encode("utf-8")).hexdigest() != self.representation_sha256:
            raise ValueError("Prepared Document text/representation SHA-256 binding is invalid")
        return self


class StudySourceBundle(RoBModel):
    schema_version: Literal["3.0.0"] = "3.0.0"
    study_key: Key
    members: tuple[PreparedDocument, ...] = Field(min_length=1)
    missing_document_roles: tuple[str, ...] = ()
    missing_document_keys: tuple[Key, ...] = ()
    bundle_sha256: Sha256

    @model_validator(mode="after")
    def bundle_binding_is_valid(self) -> StudySourceBundle:
        member_payload = [
            {
                "document_key": item.document_key,
                "source_sha256": item.source_sha256,
                "representation_sha256": item.representation_sha256,
                "document_role": item.document_role,
                "report_key": item.report_key,
                "source_path": item.source_path,
                "representation_path": item.representation_path,
                "extractor_name": item.extractor_name,
                "extractor_version": item.extractor_version,
                "extractor_config": [
                    value.model_dump(mode="json") for value in item.extractor_config
                ],
            }
            for item in self.members
        ]
        expected = hashlib.sha256(
            canonical_json(
                {
                    "members": member_payload,
                    "missing_document_roles": self.missing_document_roles,
                    "missing_document_keys": self.missing_document_keys,
                }
            )
        ).hexdigest()
        if self.bundle_sha256 != expected:
            raise ValueError("Study Source Bundle SHA-256 is invalid")
        return self


class EvidenceQuoteCandidate(RoBModel):
    document_key: Key
    quote: str = Field(min_length=1, max_length=1_000)
    purpose: str = Field(min_length=1, max_length=1_000)


class RoBSourceSpan(RoBModel):
    document_key: Key
    source_sha256: Sha256
    representation_sha256: Sha256
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    exact_text: str = Field(max_length=4_000)
    match_kind: Literal["EXACT", "NORMALIZED", "UNRESOLVED"]
    resolver_version: Literal["rob-v3-quote-resolver-1"] = "rob-v3-quote-resolver-1"
    nearest_text: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def match_binding_is_valid(self) -> RoBSourceSpan:
        if self.match_kind == "EXACT":
            if self.end_offset <= self.start_offset or not self.exact_text:
                raise ValueError("EXACT Source Span requires ordered offsets and text")
        elif self.start_offset != 0 or self.end_offset != 0 or self.exact_text:
            raise ValueError("non-exact quote diagnostic cannot publish a Source Span")
        return self


class AbsenceSearchHit(RoBModel):
    document_key: Key
    term: str = Field(min_length=1, max_length=200)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    exact_text: str = Field(min_length=1, max_length=1_000)


class AbsenceSearchRecord(RoBModel):
    probe_id: str = Field(min_length=1, max_length=200)
    probe_version: Literal["1"] = "1"
    study_key: Key
    domain: Domain
    terms: tuple[str, ...] = Field(min_length=1)
    searched_document_keys: tuple[Key, ...]
    searched_sections: tuple[str, ...]
    hit_count: int = Field(ge=0)
    hits: tuple[AbsenceSearchHit, ...]
    bundle_sha256: Sha256
    conclusion: Literal["NO_HIT", "HIT_REQUIRES_REVIEW", "SEARCH_INCOMPLETE"]

    @model_validator(mode="after")
    def search_count_is_valid(self) -> AbsenceSearchRecord:
        if self.hit_count != len(self.hits):
            raise ValueError("absence search hit count is invalid")
        if self.conclusion == "NO_HIT" and self.hits:
            raise ValueError("NO_HIT record cannot contain hits")
        if self.conclusion == "HIT_REQUIRES_REVIEW" and not self.hits:
            raise ValueError("HIT_REQUIRES_REVIEW requires hits")
        return self


class AlgorithmResult(RoBModel):
    method_fingerprint: Sha256
    domain: Domain | Literal["OVERALL"]
    input_answers_sha256: Sha256
    visited_nodes: tuple[str, ...] = Field(min_length=1)
    selected_edges: tuple[str, ...] = Field(min_length=1)
    proposed_judgment: Judgment


