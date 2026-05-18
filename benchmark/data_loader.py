"""Load ground-truth annotations from JSON files produced by process.py."""

import json
import os
import re
from pathlib import Path

from benchmark.schema import ImageAnnotation, Region


def _normalize_text(text: str) -> str:
    """Normalize escape sequences and whitespace for consistent comparison."""
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _find_image(image_dir: Path, image_id: str) -> Path | None:
    """Find image file matching an image_id, trying common extensions."""
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = image_dir / (image_id + ext)
        if candidate.exists():
            return candidate
    return None


def _get_image_size(image_path: Path) -> tuple[int, int]:
    """Get image width and height without external dependencies."""
    import struct
    # Use imghdr-style detection
    with open(image_path, "rb") as f:
        header = f.read(32)
    suffix = image_path.suffix.lower()

    if suffix in (".png"):
        # PNG: IHDR chunk at byte 16
        w, h = struct.unpack(">II", header[16:24])
        return w, h
    elif suffix in (".jpg", ".jpeg"):
        # JPEG: scan for SOF marker
        fsize = image_path.stat().st_size
        with open(image_path, "rb") as f:
            data = f.read()
        i = 2
        while i < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if 0xC0 <= marker <= 0xC3 or 0xC5 <= marker <= 0xC7 \
                    or 0xC9 <= marker <= 0xCB or 0xCD <= marker <= 0xCF:
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return 0, 0


def _is_percentage_coords(points_list: list) -> bool:
    """Check if all points are in percentage space (0-100)."""
    for pts in points_list:
        for p in pts:
            if p[0] > 100 or p[1] > 100:
                return False
    return True


def _scale_points(points: list, img_w: int, img_h: int) -> list:
    """Scale points from percentage (0-100) to absolute pixels."""
    return [[p[0] / 100.0 * img_w, p[1] / 100.0 * img_h] for p in points]


def load_gt_file(gt_path: str | Path) -> ImageAnnotation:
    """Load a single GT JSON file and return an ImageAnnotation.

    Expects JSON format from process.py: a list of dicts, each with
    "transcription", "points" (4 corners), and "label".

    Coordinates may be either absolute pixels or percentage (0-100);
    percentage coordinates are automatically scaled to pixel space.
    """
    gt_path = Path(gt_path)

    with open(gt_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Derive image_id from filename: strip "_gt" suffix
    stem = gt_path.stem
    if stem.endswith("_gt"):
        image_id = stem[:-3]
    else:
        image_id = stem

    # Find corresponding image
    image_dir = gt_path.parent.parent / "image"
    image_path = _find_image(image_dir, image_id)

    # Determine image dimensions and coordinate space
    img_w, img_h = 0, 0
    all_points = [item.get("points", []) for item in raw if len(item.get("points", [])) == 4]
    if _is_percentage_coords(all_points) and image_path:
        img_w, img_h = _get_image_size(image_path)

    regions = []
    for item in raw:
        points = item.get("points", [])
        if len(points) != 4:
            continue

        if img_w > 0 and img_h > 0:
            points = _scale_points(points, img_w, img_h)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        transcription = item.get("transcription", "")
        if item.get("label", "text") == "text":
            transcription = _normalize_text(transcription)

        region = Region(
            bbox=[min(xs), min(ys), max(xs), max(ys)],
            label=item.get("label", "text"),
            transcription=transcription,
            confidence=1.0,
        )
        regions.append(region)

    return ImageAnnotation(
        image_id=image_id,
        image_path=str(image_path) if image_path else "",
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
