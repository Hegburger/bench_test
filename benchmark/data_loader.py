"""Load ground-truth annotations from JSON files produced by process.py."""

import json
import os
from pathlib import Path

from benchmark.schema import ImageAnnotation, Region


def load_gt_file(gt_path: str | Path) -> ImageAnnotation:
    """Load a single GT JSON file and return an ImageAnnotation.

    Expects JSON format from process.py: a list of dicts, each with
    "transcription", "points" (4 corners), and "label".

    The image_id is derived from the filename (e.g. "古文字导论_165_gt.json" -> "古文字导论_165").
    """
    gt_path = Path(gt_path)

    with open(gt_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    regions = []
    for item in raw:
        points = item.get("points", [])
        if len(points) != 4:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        region = Region(
            bbox=[min(xs), min(ys), max(xs), max(ys)],
            label=item.get("label", "text"),
            transcription=item.get("transcription", ""),
            confidence=1.0,
        )
        regions.append(region)

    # Derive image_id from filename: strip "_gt" suffix
    stem = gt_path.stem  # e.g. "古文字导论_165_gt"
    if stem.endswith("_gt"):
        image_id = stem[:-3]
    else:
        image_id = stem

    # Find corresponding image
    image_name = image_id + ".png"
    image_path = gt_path.parent.parent / "image" / image_name
    if not image_path.exists():
        # Try common extensions
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = gt_path.parent.parent / "image" / (image_id + ext)
            if candidate.exists():
                image_path = candidate
                break

    return ImageAnnotation(
        image_id=image_id,
        image_path=str(image_path),
        regions=regions,
    )


def load_gt_directory(gt_dir: str | Path) -> list[ImageAnnotation]:
    """Load all GT JSON files from a directory."""
    gt_dir = Path(gt_dir)
    annotations = []

    for fname in sorted(gt_dir.glob("*_gt.json")):
        try:
            ann = load_gt_file(fname)
            if ann.regions:
                annotations.append(ann)
        except Exception as e:
            print(f"Warning: failed to load {fname}: {e}")

    return annotations
