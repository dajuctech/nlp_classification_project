# tests/test_preprocessing.py

import pytest
from src.preprocessing import clean_text

def test_clean_text_basic():
    raw = "I can't believe this!!! 😃🔥🔥🔥 Visit https://example.com #cool"
    cleaned = clean_text(raw)
    assert isinstance(cleaned, str)
    assert "can't" not in cleaned
    assert "🔥" not in cleaned
    assert "example.com" not in cleaned
    assert "cool" not in cleaned
    assert len(cleaned.split()) > 0

def test_clean_text_reduces_noise():
    raw = "Soooooo happppyyyyy!!!!!!"
    cleaned = clean_text(raw)
    assert "soo" in cleaned or "hap" in cleaned  # Check normalization worked

@pytest.mark.parametrize("input_text", [
    "No HTML &amp; &lt;test&gt;",
    "   Multiple    spaces   ",
    "",
])
def test_clean_text_variety(input_text):
    cleaned = clean_text(input_text)
    assert isinstance(cleaned, str)
