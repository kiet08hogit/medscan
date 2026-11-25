"""
Experimental evaluation framework for MedScan.

Research question
-----------------
How do different image preprocessing techniques affect medication label
recognition accuracy under varying real-world conditions?

This script does NOT invent results. It reads a ground-truth CSV and images
from dataset/, runs Methods A–F, writes metrics CSVs, and plots comparisons.

Usage (from the MedScan/ directory):
    python evaluation.py
    python evaluation.py --images dataset/images --ground-truth dataset/ground_truth.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from medication_matching import (
    load_vocabulary,
    match_medications,
    names_match,
    strengths_match,
)
from preprocessing import PREPROCESSING_METHODS, preprocess
from recognition import run_ocr
from utils import (
    character_error_rate,
    list_dataset_images,
    load_image,
    word_error_rate,
)


# Method definitions used in the research comparison.
# Method F uses EasyOCR; A–E use Tesseract.
EXPERIMENT_METHODS = [
    {
        "method_id": "A",
        "name": "Original + Tesseract",
        "preprocess": "A_original",
        "ocr": "tesseract",
    },
    {
        "method_id": "B",
        "name": "Grayscale + Tesseract",
        "preprocess": "B_grayscale",
        "ocr": "tesseract",
    },
    {
        "method_id": "C",
        "name": "Blur + Adaptive thresh + Tesseract",
        "preprocess": "C_blur_adaptive",
        "ocr": "tesseract",
    },
    {
        "method_id": "D",
        "name": "CLAHE + Otsu + Tesseract",
        "preprocess": "D_clahe_otsu",
        "ocr": "tesseract",
    },
    {
        "method_id": "E",
        "name": "Perspective + preprocess + Tesseract",
        "preprocess": "E_perspective_clahe_otsu",
        "ocr": "tesseract",
    },
    {
        "method_id": "F",
        "name": "Best preprocess + EasyOCR",
        "preprocess": "F_perspective_clahe",
        "ocr": "easyocr",
    },
]


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """Load ground-truth labels. Required columns are documented in README."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Ground-truth CSV not found: {csv_path}\n"
            "Collect images, then fill dataset/ground_truth.csv before running."
        )
    df = pd.read_csv(csv_path)
    required = {
        "filename",
        "medication_name",
        "strength",
        "expected_text",
        "condition",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ground_truth.csv missing columns: {sorted(missing)}")
    return df


def evaluate_one(
    image_path: Path,
    row: pd.Series,
    method: Dict[str, str],
    vocabulary: List[str],
) -> Dict:
    """Run one method on one image and return metric fields."""
    image = load_image(image_path)
    t0 = time.perf_counter()
    processed = preprocess(image, method=method["preprocess"])
    ocr = run_ocr(processed, engine=method["ocr"])
    elapsed = time.perf_counter() - t0

    if ocr.error:
        predicted_name = ""
        accepted = False
        match_reason = ocr.error
        strengths = []
        ocr_text = ""
    else:
        match = match_medications(ocr.text, vocabulary=vocabulary)
        predicted_name = match.top.name if match.top else ""
        accepted = match.accepted
        match_reason = match.reason
        strengths = match.extracted_strengths
        ocr_text = ocr.text

    expected_name = str(row["medication_name"])
    expected_strength = str(row["strength"]) if not pd.isna(row["strength"]) else ""
    expected_text = str(row["expected_text"]) if not pd.isna(row["expected_text"]) else ""

    name_correct = bool(predicted_name) and names_match(predicted_name, expected_name)
    strength_correct = strengths_match(strengths, expected_strength) if expected_strength else False
    incorrect_id = bool(predicted_name) and accepted and not name_correct

    return {
        "filename": image_path.name,
        "condition": row.get("condition", ""),
        "method_id": method["method_id"],
        "method_name": method["name"],
        "preprocess": method["preprocess"],
        "ocr_engine": method["ocr"],
        "expected_name": expected_name,
        "predicted_name": predicted_name,
        "match_accepted": accepted,
        "match_reason": match_reason,
        "name_correct": name_correct,
        "expected_strength": expected_strength,
        "predicted_strengths": ";".join(strengths),
        "strength_correct": strength_correct,
        "cer": character_error_rate(expected_text, ocr_text),
        "wer": word_error_rate(expected_text, ocr_text),
        "incorrect_identification": incorrect_id,
        "processing_time_sec": elapsed,
        "ocr_error": ocr.error or "",
        "ocr_text": ocr_text,
    }


def aggregate_metrics(per_image: pd.DataFrame) -> pd.DataFrame:
    """Compute method-level summary metrics."""
    rows = []
    for method_id, group in per_image.groupby("method_id"):
        n = len(group)
        rows.append(
            {
                "method_id": method_id,
                "method_name": group["method_name"].iloc[0],
                "n_images": n,
                "medication_name_accuracy": group["name_correct"].mean() if n else 0.0,
                "strength_extraction_accuracy": group["strength_correct"].mean() if n else 0.0,
                "mean_cer": group["cer"].mean() if n else 0.0,
                "mean_wer": group["wer"].mean() if n else 0.0,
                "incorrect_identification_rate": group["incorrect_identification"].mean()
                if n
                else 0.0,
                "avg_processing_time_sec": group["processing_time_sec"].mean() if n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("method_id")


def aggregate_by_condition(per_image: pd.DataFrame) -> pd.DataFrame:
    """Break down name accuracy by imaging condition and method."""
    rows = []
    for (method_id, condition), group in per_image.groupby(["method_id", "condition"]):
        rows.append(
            {
                "method_id": method_id,
                "method_name": group["method_name"].iloc[0],
                "condition": condition,
                "n_images": len(group),
                "medication_name_accuracy": group["name_correct"].mean(),
                "mean_cer": group["cer"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "method_id"])


def plot_results(summary: pd.DataFrame, graphs_dir: Path) -> List[Path]:
    """Save comparison graphs with Matplotlib. Returns written paths."""
    graphs_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    if summary.empty:
        return written

    labels = summary["method_id"] + "\n" + summary["method_name"]

    def _bar(column: str, ylabel: str, filename: str, ylim=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(summary)), summary[column], color="#2F6F8F")
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by method")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        path = graphs_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    _bar("medication_name_accuracy", "Medication name accuracy", "name_accuracy.png", (0, 1.05))
    _bar(
        "strength_extraction_accuracy",
        "Strength extraction accuracy",
        "strength_accuracy.png",
        (0, 1.05),
    )
    _bar("mean_cer", "Mean Character Error Rate (lower better)", "mean_cer.png")
    _bar("mean_wer", "Mean Word Error Rate (lower better)", "mean_wer.png")
    _bar(
        "incorrect_identification_rate",
        "Incorrect identification rate (lower better)",
        "incorrect_id_rate.png",
        (0, 1.05),
    )
    _bar(
        "avg_processing_time_sec",
        "Average processing time (seconds)",
        "avg_time.png",
    )
    return written


def run_evaluation(
    images_dir: Path,
    ground_truth_csv: Path,
    metrics_dir: Path,
    graphs_dir: Path,
    methods: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Path]:
    """
    Execute the full experiment.

    Returns paths of written artifact files. Raises if no labeled images exist
    so callers do not silently invent empty 'success' claims.
    """
    methods = methods or EXPERIMENT_METHODS
    gt = load_ground_truth(ground_truth_csv)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Extend vocabulary with names present in ground truth.
    vocabulary = load_vocabulary(gt["medication_name"].dropna().astype(str).tolist())

    available = {p.name: p for p in list_dataset_images(images_dir)}
    if not available:
        raise FileNotFoundError(
            f"No images found in {images_dir}. "
            "Add JPG/PNG files before running experiments."
        )

    records = []
    skipped = []
    for _, row in gt.iterrows():
        filename = str(row["filename"])
        if filename not in available:
            skipped.append(filename)
            continue
        for method in methods:
            # Guard against typos in method config.
            if method["preprocess"] not in PREPROCESSING_METHODS:
                raise KeyError(f"Unknown preprocess key: {method['preprocess']}")
            records.append(
                evaluate_one(available[filename], row, method, vocabulary)
            )

    if not records:
        raise RuntimeError(
            "No evaluation rows produced. Check that ground_truth filenames "
            "match files in the images directory.\n"
            f"Missing examples: {skipped[:5]}"
        )

    per_image = pd.DataFrame(records)
    summary = aggregate_metrics(per_image)
    by_condition = aggregate_by_condition(per_image)

    per_image_path = metrics_dir / "per_image_results.csv"
    summary_path = metrics_dir / "summary_by_method.csv"
    condition_path = metrics_dir / "summary_by_condition.csv"

    per_image.to_csv(per_image_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_condition.to_csv(condition_path, index=False)

    graph_paths = plot_results(summary, graphs_dir)

    print(f"Evaluated {len(per_image)} image-method pairs.")
    if skipped:
        print(f"Skipped {len(skipped)} ground-truth rows with missing images.")
    print("Summary:")
    print(summary.to_string(index=False))
    print(f"Wrote {per_image_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {condition_path}")
    for path in graph_paths:
        print(f"Wrote {path}")

    return {
        "per_image": per_image_path,
        "summary": summary_path,
        "by_condition": condition_path,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="MedScan experimental evaluation")
    parser.add_argument(
        "--images",
        type=Path,
        default=root / "dataset" / "images",
        help="Directory of test images",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=root / "dataset" / "ground_truth.csv",
        help="Ground-truth CSV path",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=root / "results" / "metrics",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=root / "results" / "graphs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        images_dir=args.images,
        ground_truth_csv=args.ground_truth,
        metrics_dir=args.metrics_dir,
        graphs_dir=args.graphs_dir,
    )
