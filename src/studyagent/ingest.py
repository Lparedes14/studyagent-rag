"""Step 1 & 2: load a PDF and split it into overlapping chunks."""
import pymupdf


def load_pdf(path: str) -> tuple[str, int]:
    """Return (full_text, page_count) for a PDF file."""
    with pymupdf.open(path) as doc:
        text = "".join(page.get_text() for page in doc)
        return text, len(doc)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into fixed-size character chunks with overlap.

    Overlap keeps sentences that straddle a boundary retrievable from
    at least one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]
