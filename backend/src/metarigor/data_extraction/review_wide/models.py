from __future__ import annotations

import hashlib
import json
from pydantic import BaseModel, ConfigDict
class ReviewModel(BaseModel):
    """Review-wide 对外与内部对象都采用封闭且不可变的 schema。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def canonical_sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


