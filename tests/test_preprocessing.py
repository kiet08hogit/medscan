"""Unit tests for OpenCV preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest

import preprocessing as pp


@pytest.fixture
def color_image() -> np.ndarray:
    """Synthetic BGR image with a bright rectangle (fake label)."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    img[80:320, 100:500] = (220, 220, 220)
    # Draw a dark "text-like" bar.
    img[150:170, 140:460] = (20, 20, 20)
    return img


def test_resize_respects_max_dimension(color_image):
    out = pp.resize_image(color_image, max_dimension=300)
    assert max(out.shape[:2]) == 300
    assert out.shape[0] / out.shape[1] == pytest.approx(
        color_image.shape[0] / color_image.shape[1], rel=1e-2
    )


def test_grayscale_shape(color_image):
    gray = pp.to_grayscale(color_image)
    assert gray.ndim == 2
    assert gray.shape == color_image.shape[:2]


def test_gaussian_blur_same_shape(color_image):
    gray = pp.to_grayscale(color_image)
    blurred = pp.gaussian_blur(gray, kernel_size=5)
    assert blurred.shape == gray.shape


def test_clahe_improves_or_changes_contrast(color_image):
    enhanced = pp.apply_clahe(color_image)
    assert enhanced.ndim == 2
    assert enhanced.dtype == np.uint8
    # CLAHE should not return an all-zero image for this fixture.
    assert enhanced.mean() > 0


def test_adaptive_and_otsu_are_binary(color_image):
    adaptive = pp.adaptive_threshold(color_image)
    otsu = pp.otsu_threshold(color_image)
    assert set(np.unique(adaptive)).issubset({0, 255})
    assert set(np.unique(otsu)).issubset({0, 255})


def test_canny_edges_not_empty(color_image):
    edges = pp.canny_edges(color_image)
    assert edges.shape == color_image.shape[:2]
    assert edges.max() == 255


def test_perspective_fallback_without_rectangle():
    # Solid gray has no package contour → must fall back to the original.
    blank = np.full((200, 200, 3), 128, dtype=np.uint8)
    warped, success = pp.perspective_transform(blank)
    assert success is False
    assert warped.shape == blank.shape
    assert np.array_equal(warped, blank)


def test_named_pipelines_run(color_image):
    for name, fn in pp.PREPROCESSING_METHODS.items():
        out = fn(color_image)
        assert out is not None
        assert out.size > 0, name


def test_visualize_steps_keys(color_image):
    steps = pp.visualize_steps(color_image)
    for key in (
        "original",
        "grayscale",
        "canny_edges",
        "otsu_threshold",
        "perspective_corrected",
    ):
        assert key in steps
