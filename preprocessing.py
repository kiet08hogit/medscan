"""
Image preprocessing pipeline for medication label OCR.

Each technique is implemented as its own function so evaluation.py can
compare combinations (grayscale only, CLAHE+Otsu, perspective correction, etc.).

Why preprocess at all?
----------------------
OCR engines work best when text is sharp, high-contrast, upright, and
relatively free of background clutter. Real packaging photos are often
angled, dim, shiny, or curved. Preprocessing tries to reduce those problems
before text recognition.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from utils import ensure_bgr, order_points


# ---------------------------------------------------------------------------
# Basic transforms
# ---------------------------------------------------------------------------

def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_dimension: Optional[int] = 1600,
) -> np.ndarray:
    """
    Resize while preserving aspect ratio.

    Large images slow OCR without always improving accuracy. Shrinking very
    large photos (e.g. phone camera 4000px+) is a practical first step.
    """
    h, w = image.shape[:2]

    if width is None and height is None and max_dimension is not None:
        longest = max(h, w)
        if longest <= max_dimension:
            return image.copy()
        scale = max_dimension / float(longest)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if width is not None and height is None:
        scale = width / float(w)
        new_w, new_h = width, int(h * scale)
    elif height is not None and width is None:
        scale = height / float(h)
        new_w, new_h = int(w * scale), height
    elif width is not None and height is not None:
        new_w, new_h = width, height
    else:
        return image.copy()

    interpolation = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR color to a single-channel grayscale image.

    Color rarely helps OCR for printed black/white label text, and grayscale
    reduces noise from colorful packaging graphics.
    """
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Smooth high-frequency noise with a Gaussian kernel.

    Light blur can help thresholding by reducing tiny speckles, but too much
    blur destroys thin letter strokes. Kernel size must be odd and positive.
    """
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    k = max(1, k)
    return cv2.GaussianBlur(image, (k, k), 0)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Unlike global histogram equalization, CLAHE enhances local contrast in
    tiles. This helps labels photographed in uneven lighting (one side bright,
    one side dark) without blowing out highlights as aggressively.
    """
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# Thresholding & edges
# ---------------------------------------------------------------------------

def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 35,
    c: int = 11,
) -> np.ndarray:
    """
    Adaptive (local) thresholding.

    Each pixel is compared to a neighborhood mean. Useful when illumination
    varies across the package. block_size must be odd.
    """
    gray = to_grayscale(image)
    b = block_size if block_size % 2 == 1 else block_size + 1
    b = max(3, b)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        b,
        c,
    )


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Otsu's global threshold.

    Automatically picks a threshold that separates foreground/background by
    maximizing between-class variance. Works well when the histogram is
    roughly bimodal (dark text on light label).
    """
    gray = to_grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def canny_edges(
    image: np.ndarray,
    low: int = 50,
    high: int = 150,
) -> np.ndarray:
    """
    Canny edge detection.

    Produces a binary edge map used to find package boundaries for perspective
    correction. The two thresholds control hysteresis: edges above `high` are
    kept, edges between `low` and `high` are kept only if connected to strong ones.
    """
    gray = to_grayscale(image)
    blurred = gaussian_blur(gray, 5)
    return cv2.Canny(blurred, low, high)


def morphological_cleanup(
    image: np.ndarray,
    operation: str = "close",
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Morphological open/close to improve text blobs.

    - Closing (dilate then erode): fills small gaps inside letters.
    - Opening (erode then dilate): removes small speckles outside letters.

    Use carefully; aggressive morphology can merge neighboring characters.
    """
    binary = image if image.ndim == 2 else to_grayscale(image)
    # If the image looks continuous-tone (not binary), threshold first.
    if len(np.unique(binary)) > 3:
        binary = otsu_threshold(binary)

    k = max(1, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    op = operation.lower().strip()
    if op == "open":
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if op == "dilate":
        return cv2.dilate(binary, kernel, iterations=iterations)
    if op == "erode":
        return cv2.erode(binary, kernel, iterations=iterations)
    # Default: close
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)


# ---------------------------------------------------------------------------
# Contours & perspective correction
# ---------------------------------------------------------------------------

def find_document_contour(
    image: np.ndarray,
    min_area_ratio: float = 0.15,
) -> Optional[np.ndarray]:
    """
    Find the largest rectangular-ish contour that might be packaging.

    Returns 4 corner points or None if no reliable rectangle is found.
    """
    edges = canny_edges(image)
    # Dilate edges slightly so broken package borders connect.
    edges = morphological_cleanup(edges, operation="dilate", kernel_size=3, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(image.shape[0] * image.shape[1])
    candidates = []

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area >= min_area_ratio * image_area:
            candidates.append((area, approx.reshape(4, 2)))

    if not candidates:
        return None

    # Prefer the largest quadrilateral.
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1].astype(np.float32)


def perspective_transform(
    image: np.ndarray,
    corners: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Warp a quadrilateral region into a frontal rectangle.

    Falls back to the original image when corners are missing or invalid.
    Returns (warped_or_original, success_flag).
    """
    if corners is None:
        corners = find_document_contour(image)

    if corners is None or len(corners) != 4:
        return image.copy(), False

    rect = order_points(corners)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 20 or max_height < 20:
        return image.copy(), False

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped, True


def draw_contours_preview(image: np.ndarray) -> np.ndarray:
    """Overlay detected package contour (if any) for visualization."""
    preview = ensure_bgr(image.copy())
    corners = find_document_contour(image)
    if corners is not None:
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    return preview


# ---------------------------------------------------------------------------
# Named pipelines used by the app and evaluation script
# ---------------------------------------------------------------------------

def pipeline_none(image: np.ndarray) -> np.ndarray:
    """Method A: original image (resized only for consistency)."""
    return resize_image(image)


def pipeline_grayscale(image: np.ndarray) -> np.ndarray:
    """Method B: resize + grayscale."""
    return to_grayscale(resize_image(image))


def pipeline_blur_adaptive(image: np.ndarray) -> np.ndarray:
    """Method C: Gaussian blur + adaptive threshold."""
    resized = resize_image(image)
    gray = to_grayscale(resized)
    blurred = gaussian_blur(gray, kernel_size=5)
    return adaptive_threshold(blurred)


def pipeline_clahe_otsu(image: np.ndarray) -> np.ndarray:
    """Method D: CLAHE + Otsu threshold."""
    resized = resize_image(image)
    enhanced = apply_clahe(resized)
    return otsu_threshold(enhanced)


def pipeline_perspective_clahe_otsu(image: np.ndarray) -> np.ndarray:
    """
    Method E: try perspective correction, then CLAHE + Otsu.

    If no reliable rectangle is found, process the original (resized) image.
    """
    resized = resize_image(image)
    corrected, _ = perspective_transform(resized)
    enhanced = apply_clahe(corrected)
    binary = otsu_threshold(enhanced)
    return morphological_cleanup(binary, operation="close", kernel_size=2)


def pipeline_best_for_easyocr(image: np.ndarray) -> np.ndarray:
    """
    Method F preprocessing: perspective correction + CLAHE (keep grayscale).

    EasyOCR often prefers continuous-tone grayscale over hard binary images,
    so we stop before Otsu here.
    """
    resized = resize_image(image)
    corrected, _ = perspective_transform(resized)
    return apply_clahe(corrected)


PREPROCESSING_METHODS = {
    "A_original": pipeline_none,
    "B_grayscale": pipeline_grayscale,
    "C_blur_adaptive": pipeline_blur_adaptive,
    "D_clahe_otsu": pipeline_clahe_otsu,
    "E_perspective_clahe_otsu": pipeline_perspective_clahe_otsu,
    "F_perspective_clahe": pipeline_best_for_easyocr,
}


def preprocess(
    image: np.ndarray,
    method: str = "D_clahe_otsu",
) -> np.ndarray:
    """Run a named preprocessing method. Unknown names fall back to D."""
    fn = PREPROCESSING_METHODS.get(method, pipeline_clahe_otsu)
    return fn(image)


def visualize_steps(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Produce intermediate images for side-by-side UI / debugging.

    Keys are human-readable step names.
    """
    resized = resize_image(image)
    gray = to_grayscale(resized)
    blurred = gaussian_blur(gray)
    clahe = apply_clahe(resized)
    adaptive = adaptive_threshold(blurred)
    otsu = otsu_threshold(clahe)
    edges = canny_edges(resized)
    corrected, success = perspective_transform(resized)
    morph = morphological_cleanup(otsu, operation="close", kernel_size=2)

    return {
        "original": resized,
        "grayscale": gray,
        "gaussian_blur": blurred,
        "clahe": clahe,
        "adaptive_threshold": adaptive,
        "otsu_threshold": otsu,
        "canny_edges": edges,
        "contour_preview": draw_contours_preview(resized),
        "perspective_corrected": corrected,
        "perspective_success": np.array([1 if success else 0]),
        "morphology": morph,
    }
