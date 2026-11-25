"""Integration checks: OCR on a synthetic label, evaluation outputs, FDA parsing."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

from evaluation import run_evaluation
from medication_lookup import lookup_medication
from medication_matching import match_medications
from recognition import best_effort_ocr, draw_ocr_boxes, extract_medication_fields, run_tesseract


def _label_image(text: str = "Ibuprofen 200 mg") -> np.ndarray:
    image = np.full((220, 720, 3), 245, dtype=np.uint8)
    cv2.putText(
        image,
        text,
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )
    return image


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_tesseract_reads_synthetic_label():
    result = run_tesseract(_label_image())
    assert result.error is None
    assert "ibuprofen" in result.text.lower()
    assert "200" in result.text
    fields = extract_medication_fields(result.text)
    assert fields["primary_strength"] == "200 mg"
    boxed = draw_ocr_boxes(_label_image(), result)
    assert boxed.shape[2] == 3


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_auto_scan_reads_synthetic_label():
    scan = best_effort_ocr(_label_image())
    assert scan.ocr.error is None
    assert "ibuprofen" in scan.ocr.text.lower()
    assert scan.image.size > 0


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_evaluation_writes_metrics_for_one_image(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "ibuprofen.png"), _label_image())
    gt = tmp_path / "ground_truth.csv"
    gt.write_text(
        "filename,medication_name,strength,expected_text,condition\n"
        "ibuprofen.png,ibuprofen,200 mg,Ibuprofen 200 mg,normal_lighting\n"
    )
    # One Tesseract method keeps the check fast (EasyOCR model download skipped).
    method = {
        "method_id": "B",
        "name": "Grayscale + Tesseract",
        "preprocess": "B_grayscale",
        "ocr": "tesseract",
    }
    paths = run_evaluation(
        images_dir=images,
        ground_truth_csv=gt,
        metrics_dir=tmp_path / "metrics",
        graphs_dir=tmp_path / "graphs",
        methods=[method],
    )
    summary = pd.read_csv(paths["summary"])
    assert summary.loc[0, "n_images"] == 1
    assert summary.loc[0, "medication_name_accuracy"] == 1.0
    assert summary.loc[0, "strength_extraction_accuracy"] == 1.0
    assert (tmp_path / "graphs" / "name_accuracy.png").exists()


def test_evaluation_refuses_empty_dataset(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    gt = tmp_path / "ground_truth.csv"
    gt.write_text(
        "filename,medication_name,strength,expected_text,condition\n"
    )
    with pytest.raises(FileNotFoundError):
        run_evaluation(
            images_dir=images,
            ground_truth_csv=gt,
            metrics_dir=tmp_path / "metrics",
            graphs_dir=tmp_path / "graphs",
            methods=[],
        )


def test_ambiguous_names_are_rejected():
    # token_set_ratio treats the shorter name as a subset of the longer one.
    result = match_medications(
        "tylenol",
        vocabulary=["tylenol", "tylenol extra"],
        min_score=70,
        ambiguity_gap=5,
    )
    assert result.accepted is False
    assert "Ambiguous" in result.reason


def test_lookup_parses_openfda_payload():
    payload = {
        "results": [
            {
                "indications_and_usage": ["Temporary relief of minor aches."],
                "warnings": ["Do not use if allergic."],
                "adverse_reactions": ["Nausea."],
                "drug_interactions": ["Ask a doctor about blood thinners."],
                "active_ingredient": ["Ibuprofen 200 mg"],
                "openfda": {
                    "brand_name": ["Advil"],
                    "generic_name": ["IBUPROFEN"],
                    "manufacturer_name": ["Example Labs"],
                },
            }
        ]
    }
    response = MagicMock(status_code=200)
    response.json.return_value = payload
    with patch("medication_lookup.requests.get", return_value=response) as get:
        info = lookup_medication("ibuprofen")
    get.assert_called()
    assert info.found is True
    assert info.brand_name == "Advil"
    assert info.generic_name == "IBUPROFEN"
    assert "aches" in info.indications_and_usage
    display = info.as_display_dict()
    assert display["Source"].startswith("openFDA")


def test_lookup_not_found():
    response = MagicMock(status_code=404)
    with patch("medication_lookup.requests.get", return_value=response):
        info = lookup_medication("not-a-real-drug-xyz")
    assert info.found is False
    assert info.error
