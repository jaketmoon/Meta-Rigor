from __future__ import annotations

"""Deterministic blind transcription of EC V2 OpenHands free-form Markdown.

The OpenHands output contract requires five explicit domain judgments, an overall
downgrade, and final certainty. Parse only those fields and rebind evidence IDs and
verbatim excerpts to the blind package; do not read gold, MR outputs, or external comparators, or infer study semantics.
"""

