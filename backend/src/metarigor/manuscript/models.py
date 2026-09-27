from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


Identifier = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,159}$")]


MethodBasis = Literal[
    "PRESPECIFIED_PROTOCOL", "FINAL_REVIEW_METHOD", "EXECUTION_ARTIFACT", "UNCLEAR_MIXED"
]


class ManuscriptModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self)).hexdigest()


def canonical_json(value: BaseModel | dict) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class SourceAnchor(ManuscriptModel):
    anchor_id: Identifier
    document_id: Identifier
    input_path: str = Field(min_length=1, max_length=10_000)
    document_sha256: Sha256
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=20_000)
    quote_sha256: Sha256
    source_representation: Literal["RAW_SOURCE", "CURATED_DERIVATION"]

    @model_validator(mode="after")
    def offsets_and_hash_are_exact(self) -> SourceAnchor:
        if self.end_offset - self.start_offset != len(self.quote):
            raise ValueError("Source Anchor offsets must match quote character length")
        if hashlib.sha256(self.quote.encode("utf-8")).hexdigest() != self.quote_sha256:
            raise ValueError("Source Anchor quote SHA-256 mismatch")
        return self


class ManuscriptReference(ManuscriptModel):
    citation_key: Identifier
    display_text: str = Field(min_length=1, max_length=4_000)
    source_anchors: tuple[SourceAnchor, ...] = Field(min_length=1, max_length=10)


class PackageIssue(ManuscriptModel):
    issue_id: Identifier
    severity: Literal["INFO", "WARNING", "ERROR"]
    category: str = Field(min_length=1, max_length=200)
    detail: str = Field(min_length=1, max_length=4_000)


class ManuscriptScope(ManuscriptModel):
    review_type: Literal["SYSTEMATIC_REVIEW_WITH_OPTIONAL_META_ANALYSIS"]
    population: str
    intervention: str
    comparator: str
    outcomes: tuple[str, ...] = Field(min_length=1, max_length=100)
    excluded_features: tuple[Literal["SUBGROUP_ANALYSIS", "META_REGRESSION"], ...]


class FactJudgeEvidence(ManuscriptModel):
    fact_id: Identifier
    relation: str = Field(min_length=1, max_length=4_000)


JudgeCriterionId = Literal[
    "FACTUAL_FIDELITY",
    "PROTOCOL_AND_CONDUCT_FIDELITY",
    "MATERIAL_COVERAGE",
    "INTERPRETATION_CALIBRATION",
    "SOURCE_TRACEABILITY",
    "CROSS_SECTION_CONSISTENCY",
    "SCIENTIFIC_ORGANIZATION",
]


