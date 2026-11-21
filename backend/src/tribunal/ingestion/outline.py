"""Topic outline from bookmarks or numbered headings."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tribunal.runtime_config import OutlineConfig

_HEADING = re.compile(r"(?i)^(?:chapter\s+)?\d+(?:\.\d+)*\.?\s+\S.{1,80}$")


@dataclass
class OutlineEntry:
    title: str
    page_start: int
    page_end: int
    source: str


def build_outline(
    toc: list[tuple[str, int]], pages: dict[int, str], cfg: OutlineConfig, page_count: int
) -> list[OutlineEntry]:
    # use_contents_page is not implemented in this build.
    found: list[tuple[str, int]] = []
    source = ""
    if cfg.use_bookmarks and toc:
        found, source = list(toc), "bookmark"
    elif cfg.use_headings:
        seen: set[str] = set()
        source = "heading"
        for page in sorted(pages):
            for line in pages[page].splitlines():
                line = line.strip()
                if _HEADING.match(line) and line.lower() not in seen:
                    seen.add(line.lower())
                    found.append((line, page))
    found = found[: cfg.max_topics]
    out = []
    for i, (title, start) in enumerate(found):
        end = found[i + 1][1] - 1 if i + 1 < len(found) else page_count
        out.append(OutlineEntry(title, start, max(end, start), source))
    return out
