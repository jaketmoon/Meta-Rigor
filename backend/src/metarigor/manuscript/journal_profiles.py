from __future__ import annotations

from pathlib import Path
PROFILE_FILENAMES = {
    "generic_prisma_2020_meta_analysis_v2": "generic_prisma_2020_meta_analysis.v2.json",
    "jama_meta_analysis_v2": "jama_meta_analysis.v2.json",
    "bmj_research_meta_analysis_v2": "bmj_research_meta_analysis.v2.json",
}


_PROFILE_ROOT = Path(__file__).with_name("journal_profiles")


def profile_path(profile_id: str, root: Path | None = None) -> Path:
    try:
        filename = PROFILE_FILENAMES[profile_id]
    except KeyError as error:
        raise ValueError(f"Unknown Manuscript Journal Profile: {profile_id}") from error
    return (root or _PROFILE_ROOT).resolve() / filename


