from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt


type KernelTask = Literal["binary_outcomes", "continuous_outcomes"]


type KernelSplit = Literal["DEV", "TEST"]


type ReferenceValue = StrictInt | StrictFloat | Literal["x", "unknown"]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BinaryArmOutput(_Contract):
    events: ReferenceValue
    group_size: ReferenceValue


class BinaryParsedOutput(_Contract):
    intervention: BinaryArmOutput
    comparator: BinaryArmOutput


class ContinuousArmOutput(_Contract):
    mean: ReferenceValue
    standard_deviation: ReferenceValue
    group_size: ReferenceValue


class ContinuousParsedOutput(_Contract):
    intervention: ContinuousArmOutput
    comparator: ContinuousArmOutput


type ParsedOutput = BinaryParsedOutput | ContinuousParsedOutput


class RawTextResponse(_Contract):
    content: str
    finish_reason: str | None = None
    requested_model: str = Field(min_length=1, max_length=500)
    provider_model: str = Field(min_length=1, max_length=500)
    # Read only by benchmark telemetry outside the Run Folder; never include in frozen Stage output.
    http_status: StrictInt | None = Field(default=None, ge=100, le=599, exclude=True)

