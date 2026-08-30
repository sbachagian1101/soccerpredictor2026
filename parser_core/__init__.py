from __future__ import annotations

"""Parser compatibility layer.

Some FootyStats friendly pages omit the literal ``Past H2H`` marker between
competition and fixture date.  The original competition regex required that
marker and therefore lost ``Club Friendlies`` labels.  This layer makes the
marker optional while preserving the existing parser API.
"""

from pathlib import Path
import importlib.util
import re
import sys

_PARSER_PATH = Path(__file__).resolve().parent.parent / "parser_core.py"
_SPEC = importlib.util.spec_from_file_location("_soccer_parser_core_legacy", _PARSER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load parser core from {_PARSER_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)
_ORIGINAL_COMPETITION_FROM_PREFIX = _LEGACY._competition_from_prefix


def _competition_from_prefix(prefix: str) -> tuple[str, str]:
    # Anchor to the end of the prefix so navigation text earlier on the page
    # cannot be mistaken for a country/competition pair. ``Past H2H`` is
    # optional because FootyStats friendly pages can omit it.
    tail = re.search(
        r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ .'-]{1,60})\s*/\s*"
        r"([A-Za-z0-9À-ÿ][A-Za-z0-9À-ÿ .&'()-]{1,80}?)"
        r"(?:\s+Past H2H)?\s*$",
        prefix,
        re.IGNORECASE,
    )
    if tail:
        country, competition = tail.group(1), tail.group(2)
        country = re.sub(r".*?Dark\s+", "", country, flags=re.IGNORECASE).strip()
        return country.strip(), competition.strip()
    return _ORIGINAL_COMPETITION_FROM_PREFIX(prefix)


_LEGACY._competition_from_prefix = _competition_from_prefix

parse_match_pages = _LEGACY.parse_match_pages
detect_focus_team = _LEGACY.detect_focus_team
to_team_perspective = _LEGACY.to_team_perspective
apply_observation_weights = _LEGACY.apply_observation_weights
parsing_diagnostics = _LEGACY.parsing_diagnostics
classify_match = _LEGACY.classify_match

__all__ = [
    "parse_match_pages",
    "detect_focus_team",
    "to_team_perspective",
    "apply_observation_weights",
    "parsing_diagnostics",
    "classify_match",
]
