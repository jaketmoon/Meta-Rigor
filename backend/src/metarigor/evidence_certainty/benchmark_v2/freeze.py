from __future__ import annotations

import re
FREEZER_ID = "metarigor-ec-review-jats-freezer"


FREEZER_VERSION = "1.1.0"


_EXPLICIT_GRADE = re.compile(
    r"\bGRADE\b|(?i:\bcertainty\b|\b(?:very\s+low|low|moderate|high)\s+CoE\b)"
)


_GRADE_PHRASES = re.compile(
    r"(?:"
    r"grading\s+of\s+recommendations\s+assessment"
    r"|summary\s+of\s+findings"
    r"|certainty[-\s]+of[-\s]+(?:the[-\s]+)?evidence"
    r"|certainty[-\s]+in[-\s]+(?:the[-\s]+)?evidence"
    r"|certainty[-\s]+assessment"
    r"|overall\s+certainty"
    r"|certainty\s+(?:was|were|ranged|rated|classified)"
    r"|evidence\s+(?:was|were)\s+(?:rated|classified|judged|assessed)"
    r"\s+(?:as\s+)?(?:very\s+low|high|moderate|low)"
    r"|(?:we|the\s+authors?)\s+(?:rated|judged|assessed)\s+(?:the\s+)?evidence"
    r"\s+(?:as\s+)?(?:very\s+low|high|moderate|low)"
    r"|confidence\s+(?:was|were)\s+(?:very\s+low|high|moderate|low)"
    r"|confidence\s+in\s+(?:the\s+)?cumulative\s+evidence"
    r"|(?:high|moderate|low|very\s+low)[- ]certainty\b"
    r"|quality\s+of\s+(?:the\s+)?evidence"
    r"|(?:level|strength)\s+of\s+(?:the\s+)?evidence"
    r"|evidence\s+(?:level|strength)"
    r")",
    re.IGNORECASE,
)


_GRADE_SYMBOLS = re.compile(r"(?:[⨁⊕◯⊖⊝]{2,}|(?:#x2A01;){2,})", re.IGNORECASE)


def _contains_grade_disclosure(value: str) -> bool:
    return bool(
        _EXPLICIT_GRADE.search(value)
        or _GRADE_PHRASES.search(value)
        or _GRADE_SYMBOLS.search(value)
    )


def candidate_exposes_grade_disclosure(value: str) -> bool:
    """Frozen-candidate leakage check for loaders and tests."""

    return _contains_grade_disclosure(value)

