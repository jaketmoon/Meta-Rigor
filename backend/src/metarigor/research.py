"""Resolve frozen experiment inputs in a checkout or an installed wheel."""

from pathlib import Path

SOURCE_PACKAGE = Path(__file__).resolve().parent
_BUNDLED_ROOT = SOURCE_PACKAGE / "_research"
RESEARCH_ROOT = _BUNDLED_ROOT if _BUNDLED_ROOT.is_dir() else SOURCE_PACKAGE.parents[2]
