"""
Optical Character Recognition (OCR) for medication labels.

Supports:
- Tesseract via pytesseract
- EasyOCR (CPU by default)

Both engines return full text plus optional bounding boxes / confidence scores
so the Streamlit UI can overlay detections on the image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import ensure_bgr, extract_strengths, normalize_text


@dataclass
class OCRBox:
    """One detected text region."""

    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class OCRResult:
    """Full OCR output for one image."""

    engine: str
    text: str
    boxes: List[OCRBox] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def normalized_text(self) -> str:
        return normalize_text(self.text)

    @property
    def strengths(self) -> List[str]:
        return extract_strengths(self.text)


# Lazy EasyOCR reader so importing this module is cheap during unit tests.
_EASYOCR_READER = None


def _get_easyocr_reader(languages: Sequence[str] = ("en",)):
    """Create (once) an EasyOCR reader that uses CPU only."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr

        # gpu=False keeps the project runnable on macOS without CUDA.
        _EASYOCR_READER = easyocr.Reader(list(languages), gpu=False)
    return _EASYOCR_READER


def _prepare_for_ocr(image: np.ndarray) -> np.ndarray:
    """Ensure the array is a contiguous uint8 image Tesseract/EasyOCR accept."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def run_tesseract(
    image: np.ndarray,
    config: str = "--oem 3 --psm 6",
) -> OCRResult:
    """
    Run Tesseract OCR.

    PSM 6 assumes a single uniform block of text, which matches many
    rectangular OTC labels. Change PSM in experiments if needed.
    """
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        return OCRResult(
            engine="tesseract",
            text="",
            error="pytesseract is not installed. Run: pip install pytesseract",
        )

    prepared = _prepare_for_ocr(image)

    try:
        data = pytesseract.image_to_data(
            prepared, output_type=Output.DICT, config=config
        )
        full_text = pytesseract.image_to_string(prepared, config=config)
    except pytesseract.TesseractNotFoundError:
        return OCRResult(
            engine="tesseract",
            text="",
            error=(
                "Tesseract binary not found. On macOS install with: "
                "brew install tesseract"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - surface OCR failures to UI
        return OCRResult(engine="tesseract", text="", error=str(exc))

    boxes: List[OCRBox] = []
    n = len(data.get("text", []))
    for i in range(n):
        raw = (data["text"][i] or "").strip()
        if not raw:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        x, y, w, h = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        boxes.append(OCRBox(text=raw, confidence=conf / 100.0, bbox=(x, y, w, h)))

    return OCRResult(engine="tesseract", text=full_text.strip(), boxes=boxes)


def run_easyocr(image: np.ndarray, min_confidence: float = 0.2) -> OCRResult:
    """
    Run EasyOCR on CPU.

    EasyOCR returns (bbox_points, text, confidence). We convert the polygon
    to an axis-aligned bounding box for simple OpenCV drawing.
    """
    prepared = _prepare_for_ocr(image)
    # EasyOCR expects RGB or grayscale; convert BGR color images.
    if prepared.ndim == 3 and prepared.shape[2] == 3:
        prepared_rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
    else:
        prepared_rgb = prepared

    try:
        reader = _get_easyocr_reader()
        detections = reader.readtext(prepared_rgb)
    except Exception as exc:  # noqa: BLE001
        return OCRResult(engine="easyocr", text="", error=str(exc))

    boxes: List[OCRBox] = []
    texts: List[str] = []
    for item in detections:
        # EasyOCR may return 3-tuples or dict-like results depending on version.
        if len(item) < 3:
            continue
        points, text, conf = item[0], item[1], float(item[2])
        if not text or conf < min_confidence:
            continue
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        boxes.append(OCRBox(text=text, confidence=conf, bbox=(x, y, w, h)))
        texts.append(text)

    return OCRResult(engine="easyocr", text=" ".join(texts).strip(), boxes=boxes)


def run_ocr(image: np.ndarray, engine: str = "tesseract") -> OCRResult:
    """Dispatch to Tesseract or EasyOCR."""
    engine = (engine or "tesseract").lower().strip()
    if engine == "easyocr":
        return run_easyocr(image)
    if engine == "auto":
        return best_effort_ocr(image).ocr
    return run_tesseract(image)


def _upscale_for_ocr(image: np.ndarray, min_side: int = 900) -> np.ndarray:
    """Enlarge a small label crop so letter strokes are tall enough for Tesseract."""
    height, width = image.shape[:2]
    short_side = min(height, width)
    if short_side >= min_side:
        return image
    scale = min(3.0, min_side / float(short_side))
    return cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )


def _center_crop(image: np.ndarray, fraction: float = 0.65) -> np.ndarray:
    """Keep the middle of the frame, where a handheld bottle usually sits."""
    height, width = image.shape[:2]
    crop_h = max(1, int(height * fraction))
    crop_w = max(1, int(width * fraction))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _sharpen(image: np.ndarray) -> np.ndarray:
    """Unsharp mask. Light sharpening helps slightly blurry label text."""
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


@dataclass
class ScanHit:
    """OCR result plus the exact image the boxes were measured on."""

    ocr: OCRResult
    image: np.ndarray
    variant: str


def best_effort_ocr(image: np.ndarray) -> ScanHit:
    """
    Try a few OCR-friendly views and keep the one that best matches a drug name.

    Binary thresholding is intentionally left out. On a color bottle photo it
    often destroys the label and leaves Tesseract with no text at all.
    """
    from medication_matching import load_vocabulary, match_medications
    from preprocessing import apply_clahe, resize_image

    vocabulary = load_vocabulary()
    base = resize_image(image, max_dimension=1200)
    center = _sharpen(_upscale_for_ocr(_center_crop(base)))
    # PSM 6 reads a text block (close-up label). PSM 11 reads scattered words
    # when the bottle is small in the frame. The center crop drops background
    # clutter, but it can also cut off a brand name at the top of a close shot,
    # so the full frame is always tried too.
    variants = [
        ("color psm6", base, "--oem 3 --psm 6"),
        ("color psm11", base, "--oem 3 --psm 11"),
        ("clahe psm6", apply_clahe(base), "--oem 3 --psm 6"),
        ("center psm11", center, "--oem 3 --psm 11"),
    ]

    best: Optional[ScanHit] = None
    best_score = -1.0
    for name, variant, config in variants:
        result = run_tesseract(variant, config=config)
        result.engine = f"tesseract ({name})"
        if result.error and best is None:
            best = ScanHit(ocr=result, image=variant, variant=name)
            continue
        match = match_medications(
            result.text, vocabulary=vocabulary, min_score=0
        )
        score = float(match.top.score) if match.top else 0.0
        if match.extracted_strengths:
            score += 3.0
        if score > best_score:
            best_score = score
            best = ScanHit(ocr=result, image=variant, variant=name)

    assert best is not None
    return best


def draw_ocr_boxes(
    image: np.ndarray,
    result: OCRResult,
    min_confidence: float = 0.0,
) -> np.ndarray:
    """Draw OCR bounding boxes and confidence scores on a copy of the image."""
    canvas = ensure_bgr(image.copy())
    for box in result.boxes:
        if box.confidence < min_confidence:
            continue
        x, y, w, h = box.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 180, 0), 2)
        label = f"{box.text[:18]} ({box.confidence:.2f})"
        cv2.putText(
            canvas,
            label,
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def extract_medication_fields(ocr_text: str) -> dict:
    """
    Lightweight field extraction from OCR text using regex heuristics.

    This is intentionally simple for a course project: we do not claim medical-
    grade parsing. Matching against a vocabulary (medication_matching.py) is
    the primary identification step.
    """
    strengths = extract_strengths(ocr_text)
    normalized = normalize_text(ocr_text)

    # Brand-like tokens are often ALL CAPS in the original OCR string.
    brand_candidates = re.findall(r"\b([A-Z][A-Za-z]{2,})\b", ocr_text)

    return {
        "raw_text": ocr_text,
        "normalized_text": normalized,
        "strengths": strengths,
        "primary_strength": strengths[0] if strengths else None,
        "brand_candidates": brand_candidates[:8],
    }
