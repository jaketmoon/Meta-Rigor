from __future__ import annotations

from .models import JudgeCriterionId
CRITERIA: tuple[tuple[JudgeCriterionId, int], ...] = (
    ("FACTUAL_FIDELITY", 25),
    ("PROTOCOL_AND_CONDUCT_FIDELITY", 15),
    ("MATERIAL_COVERAGE", 15),
    ("INTERPRETATION_CALIBRATION", 15),
    ("SOURCE_TRACEABILITY", 10),
    ("CROSS_SECTION_CONSISTENCY", 10),
    ("SCIENTIFIC_ORGANIZATION", 10),
)


RATING_ANCHORS = (
    "4: fully satisfies the criterion with no substantive issue",
    "3: generally satisfies it with only local issues that do not change research meaning",
    "2: partially satisfies it with a material omission or ambiguity requiring human correction",
    "1: mostly fails it and may mislead a reader",
    "0: critical failure, opposite/invented fact, severe protocol drift, "
    "or criterion not evaluable",
)


