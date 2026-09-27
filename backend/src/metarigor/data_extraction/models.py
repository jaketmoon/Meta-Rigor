from __future__ import annotations

import hashlib
import json
from pydantic import BaseModel, ConfigDict, Field, model_validator
class DataExtractionModel(BaseModel):
    """Data Extraction 的公开与内部 contract 均拒绝额外字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def canonical_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class SourceSpan(DataExtractionModel):
    """Document View 中由程序回取并可逐字核验的精确来源区间。"""

    span_id: str = Field(min_length=1, max_length=500)
    document_key: str = Field(min_length=1, max_length=200)
    view_path: str = Field(min_length=1, max_length=1_000)
    view_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=20_000)

    @model_validator(mode="after")
    def interval_matches_quote_length(self) -> SourceSpan:
        if self.end_char <= self.start_char:
            raise ValueError("Source Span end_char must be greater than start_char")
        if self.end_char - self.start_char != len(self.quote):
            raise ValueError("Source Span interval must equal quote length")
        return self


