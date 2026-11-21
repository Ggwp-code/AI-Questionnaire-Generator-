from tribunal.ingestion.chunking import chunk_pages


def test_chunks_slice_source():
    pages = {1: "word " * 500, 2: "short page"}
    chunks = chunk_pages(pages, 800, 100)
    for c in chunks:
        assert c.text == pages[c.page_number][c.char_start : c.char_end] and len(c.text) <= 800
    assert {c.page_number for c in chunks} == {1, 2}
    p1 = [c for c in chunks if c.page_number == 1]
    assert p1[0].char_start == 0 and p1[-1].char_end == len(pages[1])
    for a, b in zip(p1, p1[1:], strict=False):
        assert b.char_start > a.char_start and a.char_end - b.char_start <= 100


def test_chunks_drop_blank():
    assert chunk_pages({1: "   \n  "}, 800, 100) == []


def test_chunks_unicode():
    pages = {1: "日本語 " * 300}
    chunks = chunk_pages(pages, 200, 20)
    assert len(chunks) > 1
    assert all(c.text == pages[1][c.char_start : c.char_end] for c in chunks)
