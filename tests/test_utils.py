import pytest
from utils import chunk_text

def test_chunk_text_basic_split():
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)

def test_chunk_text_short_text_single_chunk():
    text = "short text"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_overlap_works():
    text = " ".join([f"word{i}" for i in range(20)])
    chunks = chunk_text(text, chunk_size=10, overlap=5)
    first_words = set(chunks[0].split())
    second_words = set(chunks[1].split())
    assert len(first_words & second_words)>0

def test_chunk_text_empty_string():
    chunks = chunk_text("", chunk_size=500, overlap=50)
    assert chunks==[] or chunks==[""]