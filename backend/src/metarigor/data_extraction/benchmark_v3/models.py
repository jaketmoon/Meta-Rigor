from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator
from metarigor.data_extraction.review_wide.models import ReviewModel
class ReviewAnalysisShell(ReviewModel):
    """Source inventory for V3 case curation only; not part of the candidate DE Run contract."""

    analysis_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,239}$")
    label: str = Field(min_length=1, max_length=2_000)
    result_type: Literal[
        "BINARY",
        "CONTINUOUS",
        "TIME_TO_EVENT",
        "ORDINAL_GIV",
        "RECURRENT_EVENT",
        "IPD",
        "NETWORK_META_ANALYSIS",
        "OTHER",
    ]
    support: Literal["M0_CORE", "SOURCE_GATED", "UNSUPPORTED_METHOD", "REFERENCE_CONFLICT"]
    blocker: str | None = Field(default=None, min_length=1, max_length=4_000)
    target_id: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,239}$")
    source_path: str | None = Field(default=None, max_length=1_000)
    source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    source_anchor: str | None = Field(default=None, min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def blocker_matches_support(self) -> ReviewAnalysisShell:
        if self.support != "M0_CORE" and self.blocker is None:
            raise ValueError("non-core analysis shell requires a precise blocker")
        source_fields = (self.source_path, self.source_sha256, self.source_anchor)
        if any(item is None for item in source_fields) and any(
            item is not None for item in source_fields
        ):
            raise ValueError("analysis source path/hash/anchor must appear together")
        return self


class ReviewCaseShell(ReviewModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,239}$")
    legacy_case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,239}$")
    legacy_split: Literal["DEVELOPMENT", "REGRESSION", "SEALED_HOLDOUT"]
    wave: Literal["WAVE_0", "WAVE_1", "WAVE_2", "CHALLENGE"]
    disposition: Literal[
        "REVIEW_CASE_CURATED",
        "CURATION_IN_PROGRESS",
        "WAITING_FOR_SOURCE_ACQUISITION",
        "UNSUPPORTED_METHOD_CHALLENGE",
        "REFERENCE_CONFLICT",
    ]
    review_source_path: str | None = Field(default=None, max_length=1_000)
    review_source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    review_source_anchor: str | None = Field(default=None, min_length=1, max_length=4_000)
    review_inventory_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    analyses: tuple[ReviewAnalysisShell, ...] = Field(min_length=1)
    disposition_reason: str = Field(min_length=1, max_length=4_000)
    candidate_request_path: str | None = Field(default=None, max_length=1_000)
    reference_package_path: str | None = Field(default=None, max_length=1_000)
    checkpoint_path: str | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def curated_case_has_complete_assets(self) -> ReviewCaseShell:
        if (self.review_source_path is None) != (self.review_source_sha256 is None):
            raise ValueError("Review shell source path and hash must appear together")
        if self.review_source_path is not None and not self.review_source_path.startswith(
            "curation-support/"
        ):
            raise ValueError("Review shell source must remain in curation-support")
        if self.review_source_path is None and (
            self.disposition != "WAITING_FOR_SOURCE_ACQUISITION"
            or not any(item.analysis_id == "review-analysis-inventory" for item in self.analyses)
        ):
            raise ValueError("missing Review source requires a waiting inventory blocker")
        assets = (self.candidate_request_path, self.reference_package_path, self.checkpoint_path)
        complete_case = self.disposition in {"REVIEW_CASE_CURATED", "REFERENCE_CONFLICT"}
        if complete_case and any(item is None for item in assets):
            raise ValueError(
                "complete or conflict Review case requires request, reference and checkpoint assets"
            )
        if complete_case and (
            self.review_source_anchor is None
            or any(
                item.target_id is not None and item.source_anchor is None for item in self.analyses
            )
        ):
            raise ValueError(
                "complete or conflict Review case requires source-grounded analysis inventory"
            )
        if complete_case and any(
            item.result_type in {"BINARY", "CONTINUOUS"} and item.target_id is None
            for item in self.analyses
        ):
            raise ValueError(
                "complete or conflict Review case must link every Binary/Continuous analysis"
            )
        if self.disposition == "WAITING_FOR_SOURCE_ACQUISITION" and not any(
            item.support == "SOURCE_GATED" for item in self.analyses
        ):
            raise ValueError("source-waiting case requires an explicit source-gated analysis")
        if self.disposition == "UNSUPPORTED_METHOD_CHALLENGE" and any(
            item.result_type in {"BINARY", "CONTINUOUS"}
            and item.support in {"M0_CORE", "SOURCE_GATED"}
            for item in self.analyses
        ):
            raise ValueError("unsupported case cannot hide a normal Binary/Continuous analysis")
        target_links = [item.target_id for item in self.analyses if item.target_id is not None]
        if len(target_links) != len(set(target_links)):
            raise ValueError("analysis shell Target links must be unique")
        return self


class ReviewBenchmarkInventory(ReviewModel):
    schema_version: Literal["3.0.0"] = "3.0.0"
    benchmark_id: Literal["metarigor-data-extraction-benchmark-v3"]
    cases: tuple[ReviewCaseShell, ...] = Field(min_length=6)

    @model_validator(mode="after")
    def cases_are_unique(self) -> ReviewBenchmarkInventory:
        ids = [case.case_id for case in self.cases]
        legacy_ids = [case.legacy_case_id for case in self.cases]
        if len(ids) != len(set(ids)) or len(legacy_ids) != len(set(legacy_ids)):
            raise ValueError("Review benchmark case and legacy ids must be unique")
        return self

