import pytest

from studyagent.ingest import chunk_text


def test_chunk_sizes_and_overlap():
    text = "a" * 2500
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert all(len(c) <= 1000 for c in chunks)
    # consecutive chunks share `overlap` characters
    assert chunks[0][-200:] == chunks[1][:200]


def test_covers_whole_text():
    text = "".join(str(i % 10) for i in range(3333))
    chunks = chunk_text(text, 1000, 200)
    assert chunks[0].startswith(text[:50])
    assert chunks[-1].endswith(text[-50:])


def test_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)
