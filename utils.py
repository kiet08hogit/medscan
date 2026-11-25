"""
Shared helper functions for MedScan.

Keep small utilities here so preprocessing, OCR, matching, and evaluation
modules stay focused on their own jobs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Strength patterns commonly printed on OTC / Rx labels.
# Use (?!\w) instead of a trailing \b so units like "%" still match
# ( "%" is non-word, so \b after it often fails before spaces ).
STRENGTH_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|ml|mL|iu|IU|g|%)(?!\w)",
    re.IGNORECASE,
)

# Characters OCR often confuses; used lightly during normalization.
OCR_CONFUSION_MAP = str.maketrans(
    {
        "|": "I",
        "¡": "i",
        "€": "e",
        "£": "E",
    }
)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk in BGR format (OpenCV default)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Use JPG, JPEG, or PNG."
        )

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def list_dataset_images(directory: str | Path) -> List[Path]:
    """Return sorted image paths from a dataset folder."""
    directory = Path(directory)
    if not directory.exists():
        return []
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(files)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB for Matplotlib / Streamlit display."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Ensure an image array is 3-channel BGR for drawing helpers."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def normalize_text(text: str) -> str:
    """
    Clean OCR output for matching.

    Steps:
    1. Fix a few common OCR symbol substitutions.
    2. Lowercase everything.
    3. Replace punctuation with spaces.
    4. Collapse repeated whitespace.
    """
    if not text:
        return ""

    cleaned = text.translate(OCR_CONFUSION_MAP)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9.%\s-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_strengths(text: str) -> List[str]:
    """
    Pull dose / strength phrases such as '200 mg' or '0.5%'.

    Returns a list of normalized strings like '200 mg'.
    """
    if not text:
        return []

    matches = STRENGTH_PATTERN.findall(text)
    strengths = []
    for amount, unit in matches:
        unit_norm = unit.lower().replace("µg", "mcg").replace("ug", "mcg")
        if unit_norm == "ml":
            unit_norm = "ml"
        strengths.append(f"{amount} {unit_norm}")
    return strengths


def extract_candidate_tokens(text: str, min_len: int = 3) -> List[str]:
    """Split normalized OCR text into tokens useful for medication matching."""
    normalized = normalize_text(text)
    tokens = []
    for token in normalized.split():
        # Skip pure numbers / units that are not drug names.
        if STRENGTH_PATTERN.fullmatch(token):
            continue
        if token.isdigit():
            continue
        if len(token) < min_len:
            continue
        tokens.append(token)
    return tokens


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four rectangle corner points as:
    top-left, top-right, bottom-right, bottom-left.
    """
    pts = np.asarray(pts, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest x+y; bottom-right has largest x+y.
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    # Top-right has smallest y-x; bottom-left has largest y-x.
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate (CER) = edit_distance / len(reference).

    Lower is better. Returns 1.0 when the reference is empty and hypothesis
    is not, and 0.0 when both are empty.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    distance = _levenshtein(ref, hyp)
    return distance / max(len(ref), 1)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word Error Rate (WER) using token-level Levenshtein distance."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0
    distance = _levenshtein(ref_words, hyp_words)
    return distance / max(len(ref_words), 1)


def _levenshtein(a: Iterable, b: Iterable) -> int:
    """Classic Levenshtein edit distance for sequences (chars or words)."""
    a_list = list(a)
    b_list = list(b)
    n, m = len(a_list), len(b_list)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i, ca in enumerate(a_list, start=1):
        curr = [i]
        for j, cb in enumerate(b_list, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[m]


def safe_filename(name: str) -> str:
    """Make a string safe for use as a filename stem."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "untitled"


def truncate(text: str, max_len: int = 400) -> str:
    """Shorten long API / OCR strings for UI display."""
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def first_nonempty(values: Iterable[Optional[str]]) -> Optional[str]:
    """Return the first non-empty string from an iterable."""
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return None
