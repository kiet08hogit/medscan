"""
MedScan — lightweight Streamlit demonstration UI.

Focus: show the computer vision / OCR pipeline clearly.
This is a class project demo, not a clinical tool.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from medication_lookup import DISCLAIMER, lookup_medication
from medication_matching import load_vocabulary, match_medications
from preprocessing import PREPROCESSING_METHODS, preprocess, visualize_steps
from recognition import best_effort_ocr, draw_ocr_boxes, extract_medication_fields, run_ocr
from utils import bgr_to_rgb, list_dataset_images, load_image


ROOT = Path(__file__).resolve().parent
DATASET_IMAGES = ROOT / "dataset" / "images"

st.set_page_config(page_title="MedScan", layout="wide")

st.title("MedScan")
st.caption("CS 415 Computer Vision — medication packaging OCR demo")
st.info(DISCLAIMER)


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    return image


def show_image(image: np.ndarray, caption: str) -> None:
    st.image(bgr_to_rgb(image), caption=caption, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Input")
input_mode = st.sidebar.radio(
    "Image source",
    ["Upload", "Dataset folder"],
)

method_keys = ["auto"] + list(PREPROCESSING_METHODS.keys())
preprocess_method = st.sidebar.selectbox(
    "Scan mode",
    method_keys,
    index=0,
    key="scan_mode",
    help="auto tries the color photo and a center crop, then keeps the read that best matches a medication. The lettered modes are the research comparisons.",
)
ocr_engine = st.sidebar.selectbox(
    "OCR engine",
    ["tesseract", "easyocr"],
    help="Ignored while Scan mode is auto. Auto always uses Tesseract.",
)
min_score = st.sidebar.slider("Minimum match score", 50, 95, 70)
show_steps = st.sidebar.checkbox("Show intermediate CV steps", value=True)

image = None
image_label = ""

if input_mode == "Upload":
    uploaded = st.sidebar.file_uploader(
        "Medication image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is not None:
        image = decode_uploaded_image(uploaded.getvalue())
        image_label = uploaded.name

else:
    dataset_files = list_dataset_images(DATASET_IMAGES)
    if not dataset_files:
        st.sidebar.warning(
            f"No images in {DATASET_IMAGES}. Add JPG/PNG files to use this mode."
        )
    else:
        choice = st.sidebar.selectbox(
            "Dataset image",
            dataset_files,
            format_func=lambda p: p.name,
        )
        if choice is not None:
            image = load_image(choice)
            image_label = choice.name


if image is None:
    st.write(
        "Provide an image using the sidebar to run the MedScan pipeline. "
        "For research experiments, collect a labeled dataset and run "
        "`python evaluation.py`."
    )
else:
    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------
    with st.spinner("Scanning the label..."):
        steps = visualize_steps(image)
        if preprocess_method == "auto":
            scan = best_effort_ocr(image)
            processed = scan.image
            ocr_result = scan.ocr
            processed_caption = f"Auto scan ({scan.variant})"
        else:
            processed = preprocess(image, method=preprocess_method)
            ocr_result = run_ocr(processed, engine=ocr_engine)
            processed_caption = f"Processed ({preprocess_method})"
        boxed = draw_ocr_boxes(
            processed
            if processed.ndim == 3
            else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR),
            ocr_result,
        )
        fields = extract_medication_fields(ocr_result.text)
        vocabulary = load_vocabulary()
        match = match_medications(
            ocr_result.text,
            vocabulary=vocabulary,
            min_score=float(min_score),
        )

    col1, col2 = st.columns(2)
    with col1:
        show_image(image, f"Original — {image_label}")
    with col2:
        show_image(processed, processed_caption)

    if show_steps:
        st.subheader("Intermediate processing")
        step_names = [
            "grayscale",
            "canny_edges",
            "adaptive_threshold",
            "otsu_threshold",
            "perspective_corrected",
            "morphology",
        ]
        cols = st.columns(3)
        for i, name in enumerate(step_names):
            with cols[i % 3]:
                show_image(steps[name], name.replace("_", " "))

    st.subheader("OCR output")
    if ocr_result.error:
        st.error(ocr_result.error)
    else:
        show_image(boxed, f"{ocr_result.engine} detections")
        st.text_area("Detected text", ocr_result.text or "(empty)", height=140)
        st.write(
            {
                "engine": ocr_result.engine,
                "primary_strength": fields["primary_strength"],
                "all_strengths": fields["strengths"],
                "brand_candidates": fields["brand_candidates"],
                "boxes": len(ocr_result.boxes),
            }
        )
        if ocr_result.boxes:
            st.dataframe(
                [
                    {
                        "text": b.text,
                        "confidence": round(b.confidence, 3),
                        "bbox_xywh": b.bbox,
                    }
                    for b in ocr_result.boxes
                ],
                use_container_width=True,
            )

    st.subheader("Medication candidates")
    st.write(match.reason)
    if not match.candidates:
        st.warning(
            "No letters were read from the label. Hold the front of the bottle closer, "
            "fill the center of the frame, and keep Scan mode on auto."
        )
    else:
        st.dataframe(
            [
                {
                    "medication": c.name,
                    "score": round(c.score, 1),
                    "matched_query": c.matched_query,
                }
                for c in match.candidates
            ],
            use_container_width=True,
        )

    candidate_names = [c.name for c in match.candidates]
    selection = st.selectbox(
        "Confirm medication before lookup",
        options=["(none — do not look up)"] + candidate_names,
        index=0 if not match.accepted else 1,
    )

    if selection != "(none — do not look up)":
        if st.button("Retrieve openFDA information", type="primary"):
            with st.spinner("Querying openFDA..."):
                info = lookup_medication(selection)
            if not info.found:
                st.error(info.error or "Medication not found in openFDA.")
            else:
                st.success(f"Found label information for '{selection}'.")
                st.json(info.as_display_dict())
                st.caption(info.source)
                st.warning(info.disclaimer)
    elif match.candidates and not match.accepted:
        st.info(
            "Automatic acceptance was rejected due to low confidence or ambiguity. "
            "You may still confirm a candidate manually above."
        )
