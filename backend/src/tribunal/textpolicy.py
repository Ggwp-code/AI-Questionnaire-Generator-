"""Plain-text policy: remove emoji while preserving mathematics, units, accents and scripts.

The check works on extended grapheme clusters, so multi-code-point emoji (skin tone
modifiers, ZWJ families, flags, tag sequences, keycaps) are handled as one unit and
combining accents on ordinary letters are never split.

Rules, applied per cluster:
1. A cluster whose base character is on the preserve list keeps its base character
   (an emoji-style variation selector is dropped). This protects symbols that the
   Unicode emoji data also lists, such as arrows, card suits, gender signs, the
   copyright sign and the digits used by keycaps.
2. A cluster that contains an emoji-presentation character, a skin tone modifier,
   a regional indicator, a tag character, a keycap mark or an emoji variation
   selector is removed.
3. A cluster whose base character is Extended_Pictographic is removed.
4. Everything else is kept unchanged. Mathematical operators (U+2200 block),
   superscripts, the minus sign, degree and micro signs, letters with accents and
   non-Latin scripts are not emoji and are never touched.

The preserve list is supplied by runtime configuration (text_policy.preserve_codepoints).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import regex

_CLUSTER = regex.compile(r"\X")
_EMOJI_MARKERS = regex.compile(
    r"[\p{Emoji_Presentation}\p{Emoji_Modifier}\p{Regional_Indicator}\U000E0020-\U000E007F\u20E3\uFE0F]"
)
_PICTOGRAPHIC = regex.compile(r"\p{Extended_Pictographic}")
_VS16 = "\ufe0f"
_KEYCAP = "\u20e3"
_ZWJ = "\u200d"


def parse_codepoints(entries: Iterable[str]) -> frozenset[int]:
    """Parse entries like '00A9' or '2194-2199' into a set of code points."""
    points: set[int] = set()
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if "-" in entry:
            start, end = entry.split("-", 1)
            lo, hi = int(start, 16), int(end, 16)
            if hi < lo:
                raise ValueError(f"invalid code point range '{entry}'")
            points.update(range(lo, hi + 1))
        else:
            points.add(int(entry, 16))
    return frozenset(points)


@dataclass(frozen=True)
class Finding:
    index: int
    cluster: str

    @property
    def codepoints(self) -> str:
        return " ".join(f"U+{ord(c):04X}" for c in self.cluster)


class TextPolicy:
    def __init__(self, preserve: Iterable[str]):
        self.preserve = parse_codepoints(preserve)

    def _classify(self, cluster: str) -> str | None:
        """Return the replacement for a cluster, or None to keep it unchanged."""
        base = cluster[0]
        if ord(base) in self.preserve:
            # Keycap sequences and emoji-style variants collapse to the base symbol.
            if _VS16 in cluster or _KEYCAP in cluster:
                return base + "".join(c for c in cluster[1:] if c not in (_VS16, _KEYCAP, _ZWJ))
            return None
        if _EMOJI_MARKERS.search(cluster) or _PICTOGRAPHIC.match(base):
            return ""
        return None

    def findings(self, text: str) -> list[Finding]:
        found: list[Finding] = []
        pos = 0
        for match in _CLUSTER.finditer(text):
            cluster = match.group(0)
            replacement = self._classify(cluster)
            if replacement is not None and replacement != cluster:
                found.append(Finding(index=pos, cluster=cluster))
            pos = match.end()
        return found

    def clean(self, text: str) -> tuple[str, int]:
        """Return (cleaned_text, number_of_clusters_changed)."""
        if not text:
            return text, 0
        out: list[str] = []
        changed = 0
        for match in _CLUSTER.finditer(text):
            cluster = match.group(0)
            replacement = self._classify(cluster)
            if replacement is None:
                out.append(cluster)
            else:
                changed += 1
                out.append(replacement)
        cleaned = "".join(out)
        if changed:
            # Removing an emoji can leave doubled spaces inside a line; collapse only those.
            cleaned = regex.sub(r"(?<=\S) {2,}(?=\S)", " ", cleaned)
            cleaned = regex.sub(r"[ \t]+$", "", cleaned, flags=regex.MULTILINE)
        return cleaned, changed

    def clean_value(self, value):
        """Recursively clean strings inside dicts and lists. Returns (value, changed_count)."""
        if isinstance(value, str):
            return self.clean(value)
        if isinstance(value, list):
            total = 0
            items = []
            for item in value:
                cleaned, n = self.clean_value(item)
                items.append(cleaned)
                total += n
            return items, total
        if isinstance(value, dict):
            total = 0
            result = {}
            for key, item in value.items():
                cleaned, n = self.clean_value(item)
                result[key] = cleaned
                total += n
            return result, total
        return value, 0
