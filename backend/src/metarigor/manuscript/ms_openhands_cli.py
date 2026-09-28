from __future__ import annotations

"""OpenHands CLI natural-output external control experiment for the MS generic ten-case benchmark.

Reuse Data Extraction's sealed, isolated OpenHands 1.16.0 runtime. Each case sees only
a Manuscript input package byte-identical to the Codex CLI control. Preserve original failures;
write user-authorized reruns as separate attempts, selected for later analysis with a ``rerun`` flag.
"""


SURFACE_ROLES = {
    "MANUSCRIPT",
    "TABLES",
    "SUPPLEMENT",
    "REFERENCES",
    "PRISMA_FLOW",
    "FOREST_PLOT",
}

