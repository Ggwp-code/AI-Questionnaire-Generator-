from tribunal.ingestion.outline import build_outline
from tribunal.runtime_config import OutlineConfig


def _cfg(**kw):
    base = dict(use_bookmarks=True, use_contents_page=False, use_headings=True, contents_search_pages=5, max_topics=40)
    return OutlineConfig(**{**base, **kw})


def test_outline_bookmarks():
    out = build_outline([("Trees", 1), ("Regression", 3)], {}, _cfg(), 4)
    assert [(e.title, e.page_start, e.page_end, e.source) for e in out] == [
        ("Trees", 1, 2, "bookmark"),
        ("Regression", 3, 4, "bookmark"),
    ]


def test_outline_headings():
    pages = {1: "1 Alpha Topic\ntext", 2: "more", 3: "2 Beta Topic\ntext"}
    out = build_outline([], pages, _cfg(), 3)
    assert [e.title for e in out] == ["1 Alpha Topic", "2 Beta Topic"]
    assert [(e.page_start, e.page_end) for e in out] == [(1, 2), (3, 3)]
    assert {e.source for e in out} == {"heading"}


def test_outline_none():
    assert build_outline([], {1: "plain prose here"}, _cfg(), 1) == []


def test_outline_max_topics():
    pages = {i: f"{i} Heading number {i}" for i in range(1, 31)}
    assert len(build_outline([], pages, _cfg(max_topics=5), 30)) == 5
