"""Tests for text normalization, strength extraction, and fuzzy matching."""

from __future__ import annotations

import pytest

from medication_matching import match_medications, names_match, strengths_match
from utils import (
    character_error_rate,
    extract_strengths,
    normalize_text,
    word_error_rate,
)


def test_normalize_text_basic():
    assert normalize_text("  Ibuprofen!!! 200mg ") == "ibuprofen 200mg"
    assert "tylenol" in normalize_text("TYLENOL")


def test_extract_strengths():
    text = "Ibuprofen Tablets 200 mg Pain Reliever"
    assert extract_strengths(text) == ["200 mg"]
    text2 = "Use 0.5% cream twice daily; also 10 mg capsule"
    strengths = extract_strengths(text2)
    assert any("0.5" in s for s in strengths)
    assert any(s.startswith("10") for s in strengths)


def test_match_ibuprofen_with_ocr_noise():
    ocr = "lbuprofen Tablets 200 mg"  # I -> l common OCR slip
    result = match_medications(ocr, min_score=70)
    assert result.candidates
    assert "ibuprofen" in result.candidates[0].name.lower()


def test_reject_low_confidence_gibberish():
    result = match_medications("xqz vvv nnn", min_score=80)
    assert result.accepted is False


def test_names_and_strengths_match_helpers():
    assert names_match("Ibuprofen", "ibuprofen")
    assert strengths_match(["200 mg"], "200 mg")
    assert not strengths_match(["200 mg"], "500 mg")


def test_cer_wer_identical():
    assert character_error_rate("hello world", "hello world") == 0.0
    assert word_error_rate("hello world", "hello world") == 0.0


def test_cer_wer_different():
    # One substitution in three characters → CER = 1/3
    assert character_error_rate("abc", "axc") == pytest.approx(1 / 3)
    assert word_error_rate("one two", "one three") > 0
