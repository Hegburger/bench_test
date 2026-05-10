"""Main evaluation pipeline.

Orchestrates matching, metric computation, and result aggregation for
a single image or a full dataset.
"""

from benchmark.config import EvalConfig, TEXTUAL_LABELS
from benchmark.matcher import match_regions
from benchmark.metrics import (
    char_error_rate,
    liding_score,
    detection_metrics,
    classification_accuracy,
    composite_score,
)
from benchmark.schema import ImageAnnotation, Region


def evaluate_image(
    pred: ImageAnnotation,
    gt: ImageAnnotation,
    config: EvalConfig | None = None,
) -> dict:
    """Evaluate a single predicted annotation against ground truth.

    Args:
        pred: Model prediction (ImageAnnotation).
        gt: Ground truth (ImageAnnotation).
        config: Evaluation configuration.

    Returns:
        Dict with detection, classification, recognition, and composite scores.
    """
    if config is None:
        config = EvalConfig()

    # ── Matching ──
    match_result = match_regions(
        pred.regions, gt.regions, iou_threshold=config.iou_threshold,
    )

    matches = match_result["matches"]
    per_label = match_result["per_label"]

    # Build (pred_region, gt_region) pairs for matched indices
    matched_pairs: list[tuple[Region, Region]] = []
    for pred_idx, gt_idx, iou in matches:
        matched_pairs.append((pred.regions[pred_idx], gt.regions[gt_idx]))

    # ── Detection metrics ──
    detection = {}
    detection_f1 = {}
    for label in ("text", "liding", "oracle", "picture"):
        pinfo = per_label.get(label, {"num_pred": 0, "num_gt": 0, "matches": []})
        dm = detection_metrics(
            num_pred=pinfo["num_pred"],
            num_gt=pinfo["num_gt"],
            num_matched=len(pinfo["matches"]),
        )
        detection[label] = dm
        detection_f1[label] = dm["f1"]

    # Overall detection
    total_matched = len(matches)
    total_pred = len(pred.regions)
    total_gt = len(gt.regions)
    detection["overall"] = detection_metrics(total_pred, total_gt, total_matched)

    # ── Classification ──
    classification = classification_accuracy(matched_pairs)

    # ── Recognition (per matched pair) ──
    recognition = {label: {"scores": [], "num_pairs": 0} for label in TEXTUAL_LABELS}
    recognition_scores = {}  # label -> mean score (0=perfect, 1=worst)

    for pred_r, gt_r in matched_pairs:
        label = gt_r.label
        if label not in TEXTUAL_LABELS:
            continue
        if not gt_r.transcription.strip():
            continue

        if label == "text":
            score = char_error_rate(gt_r.transcription, pred_r.transcription,
                                    case_sensitive=config.text_case_sensitive)
            recognition[label]["scores"].append(score)
        elif label == "liding":
            result = liding_score(gt_r.transcription, pred_r.transcription, config)
            recognition[label]["scores"].append({
                "string_ned": result["string_ned"],
                "tree_ted": result.get("tree_ted"),
                "combined": result["combined"],
            })

    for label in TEXTUAL_LABELS:
        recs = recognition[label]["scores"]
        if recs:
            if label == "text":
                recognition[label]["mean"] = sum(recs) / len(recs)
            else:
                # liding: average the combined scores
                recognition[label]["mean"] = sum(
                    r["combined"] if isinstance(r, dict) else r for r in recs
                ) / len(recs)
            recognition[label]["num_pairs"] = len(recs)
            recognition_scores[label] = recognition[label]["mean"]
        else:
            recognition[label]["mean"] = 0.0
            recognition_scores[label] = 0.0

    # ── Composite ──
    composite = composite_score(
        detection_f1_by_label=detection_f1,
        recognition_score_by_label=recognition_scores,
        classification_acc=classification["accuracy"],
        config=config,
    )

    return {
        "image_id": gt.image_id,
        "detection": detection,
        "classification": classification,
        "recognition": recognition,
        "composite": composite,
    }


def evaluate_dataset(
    predictions: dict[str, ImageAnnotation],
    ground_truths: dict[str, ImageAnnotation],
    config: EvalConfig | None = None,
) -> dict:
    """Evaluate a full dataset.

    Args:
        predictions: {image_id: ImageAnnotation} from model.
        ground_truths: {image_id: ImageAnnotation} from GT.
        config: Evaluation configuration.

    Returns:
        Dict with per-image results and aggregated summary.
    """
    if config is None:
        config = EvalConfig()

    per_image = {}
    for image_id, gt in ground_truths.items():
        pred = predictions.get(image_id)
        if pred is None:
            per_image[image_id] = {"error": "no prediction found"}
            continue
        per_image[image_id] = evaluate_image(pred, gt, config)

    # ── Aggregate summary ──
    summary = _aggregate(per_image, config)
    summary["num_images"] = len(ground_truths)
    summary["num_evaluated"] = len([r for r in per_image.values() if "error" not in r])

    return {
        "per_image": per_image,
        "summary": summary,
        "config": {
            "iou_threshold": config.iou_threshold,
            "liding_use_tree_edit": config.liding_use_tree_edit,
        },
    }


def _aggregate(per_image: dict, config: EvalConfig) -> dict:
    """Aggregate per-image results into a summary."""
    valid = [r for r in per_image.values() if "error" not in r]
    if not valid:
        return {}

    n = len(valid)

    def _mean(key_path: str) -> float:
        vals = []
        for r in valid:
            v = r
            for k in key_path.split("."):
                v = v.get(k, {}) if isinstance(v, dict) else 0
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else 0.0

    # Detection averages
    det_summary = {}
    for label in ("text", "liding", "oracle", "picture", "overall"):
        det_summary[label] = {
            "mean_precision": _mean(f"detection.{label}.precision"),
            "mean_recall": _mean(f"detection.{label}.recall"),
            "mean_f1": _mean(f"detection.{label}.f1"),
        }

    # Recognition averages
    rec_summary = {}
    for label in TEXTUAL_LABELS:
        rec_scores = []
        for r in valid:
            rec = r.get("recognition", {}).get(label, {})
            mean_val = rec.get("mean", 0.0)
            if rec.get("num_pairs", 0) > 0:
                rec_scores.append(mean_val)
        rec_summary[label] = {
            "mean_error_rate": sum(rec_scores) / len(rec_scores) if rec_scores else 0.0,
        }

    # Classification
    class_acc = _mean("classification.accuracy")

    # Composite
    composites = [r["composite"] for r in valid]

    return {
        "detection": det_summary,
        "recognition": rec_summary,
        "classification_accuracy": class_acc,
        "mean_composite": sum(composites) / len(composites) if composites else 0.0,
    }