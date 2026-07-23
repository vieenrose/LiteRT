"""Simplified -> Traditional conversion for display.

The model emits SIMPLIFIED Chinese regardless of the speech being Taiwanese --
that is a property of the base weights, not of this pipeline (verified: the
genuine PyTorch reference output is equally Simplified). Traditional output has
always been a post-processing step, never something the weights do.

Uses OpenCC `s2t` (character-level Simplified -> Traditional).

Why not the more aggressive profiles:
  * `s2twp` (phrase-level, Taiwan idiom) gives real vocabulary wins --
    軟件->軟體, 網絡->網路, 信息->資訊 -- but CORRUPTS domain proper nouns.
    Measured on this project's own zh golden clip, all 4 of its differences vs
    s2tw were corruptions: 高端疫苗 -> 高階疫苗 (Medigen, the vaccine that
    session is entirely about) and 程序委員會 -> 程式委員會. An earlier audit
    found 80/400 sampled segments affected.
  * `s2tw` adds Taiwan character variants on top of s2t (e.g. 爲 -> 為).
    s2t alone leaves the mainland variant 爲 in place.

s2t is the conservative choice: it cannot substitute words, so it cannot mangle
proper nouns. If Taiwan character variants matter later, s2tw is the safe
upgrade (still phrase-substitution-free); s2twp needs a protected-term list.
"""
from __future__ import annotations

import re

try:
    import opencc
    _CONV = opencc.OpenCC("s2t")
except Exception as _e:  # noqa: BLE001
    # Do NOT fail silently. A missing opencc once shipped to production looking
    # fine: text passed through unconverted, and a spot-check on 高端 (identical
    # in both scripts) gave a false pass. Surface it loudly at import; callers
    # can still run, but the log says why output is Simplified.
    import sys
    print(f"[zhtw] WARNING: OpenCC unavailable ({_e!r}) -- Chinese output will "
          f"NOT be converted to Traditional. Add "
          f"'opencc-python-reimplemented' to requirements.", file=sys.stderr, flush=True)
    _CONV = None


def available() -> bool:
    """True if conversion is actually active (surfaced via /api/health)."""
    return _CONV is not None

# Timestamp/speaker markers are structural output and must never be converted.
_STRUCT = re.compile(r"(\[\d+(?:\.\d+)?\]|\[S\d+\])")


def to_traditional(text: str) -> str:
    """Convert model output to Traditional Chinese, leaving [ts] and [Sxx]
    markers byte-identical. Non-Chinese text is unaffected by OpenCC, so this
    is safe to run unconditionally on English output too."""
    if _CONV is None or not text:
        return text
    parts = _STRUCT.split(text)
    return "".join(p if (not p or _STRUCT.fullmatch(p)) else _CONV.convert(p)
                   for p in parts)
