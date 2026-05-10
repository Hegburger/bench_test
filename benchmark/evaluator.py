"""Main evaluation pipeline.

Orchestrates matching, metric computation, and result aggregation for
a single image or a full dataset.
"""

from benchmark.config import EvalConfig, TEXTUAL_LABELS
from benchmark.matcher import match_regions, intersection_area
from benchmark.metrics import (
    char_error_rate,
    liding_score,
    substring_cer,
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
        pred.regions, gt.regions,
        iou_threshold=config.iou_threshold,
        containment_threshold=config.containment_threshold,
    )

    matches = match_result["matches"]
    per_label = match_result["per_label"]

    # Track matched GT/Pred indices for coverage recognition
    matched_pred = {pred_idx for pred_idx, _, _ in matches}
    matched_gt = {gt_idx for _, gt_idx, _ in matches}

    # ── Debug: per-pair detail tracking ──
    debug_pairs = [] if config.debug else None

    # Build (pred_region, gt_region) pairs for matched indices
    matched_pairs: list[tuple[Region, Region]] = []
    for pred_idx, gt_idx, match_score in matches:
        pred_r = pred.regions[pred_idx]
        gt_r = gt.regions[gt_idx]
        matched_pairs.append((pred_r, gt_r))

        if debug_pairs is not None:
            # Determine if this was IOU or containment match
            from benchmark.matcher import compute_iou, compute_containment
            iou = compute_iou(pred_r, gt_r)
            p_in_g, g_in_p = compute_containment(pred_r, gt_r)
            if iou >= config.iou_threshold:
                method = "iou"
                method_score = iou
            else:
                method = "containment"
                method_score = max(g_in_p, p_in_g)

            debug_pairs.append({
                "type": "matched",
                "pred_idx": pred_idx,
                "gt_idx": gt_idx,
                "pred_label": pred_r.label,
                "gt_label": gt_r.label,
                "pred_bbox": pred_r.bbox,
                "gt_bbox": gt_r.bbox,
                "pred_text": pred_r.transcription[:80],
                "gt_text": gt_r.transcription[:80],
                "match_method": method,
                "match_score": round(match_score, 4),
            })

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

    # Index matched debug_pairs by (pred_idx, gt_idx) for score annotation
    _dp_index = {}
    if debug_pairs is not None:
        for dp in debug_pairs:
            _dp_index[(dp["pred_idx"], dp["gt_idx"])] = dp

    for idx, (pred_r, gt_r) in enumerate(matched_pairs):
        label = gt_r.label
        if label not in TEXTUAL_LABELS:
            continue
        if not gt_r.transcription.strip():
            continue
        # Only evaluate recognition on same-label pairs
        if pred_r.label != label:
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
            score = result["combined"]

        # Annotate debug pair
        if debug_pairs is not None:
            _, gt_idx, _ = matches[idx]
            pred_idx = matches[idx][0]
            key = (pred_idx, gt_idx)
            if key in _dp_index:
                _dp_index[key]["rec_score"] = round(score, 4) if isinstance(score, (int, float)) else score

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

    # ── Coverage-based recognition for unmatched GT ──
    # Text regions inside layout-level blocks may have failed IOU matching
    # due to granularity mismatch but still have valid OCR to evaluate.
    coverage_rec = {label: {"scores": [], "num_pairs": 0} for label in TEXTUAL_LABELS}

    for j, gt_r in enumerate(gt.regions):
        if j in matched_gt:
            continue
        if gt_r.label not in TEXTUAL_LABELS:
            continue
        if not gt_r.transcription.strip():
            continue

        # Find best-covering prediction for this GT
        best_cover = 0.0
        best_pred = None
        best_pred_idx = -1
        for pi, pred_r in enumerate(pred.regions):
            if pred_r.label != gt_r.label:
                continue
            inter = intersection_area(gt_r.bbox, pred_r.bbox)
            if inter <= 0:
                continue
            cover = inter / gt_r.area
            if cover > best_cover:
                best_cover = cover
                best_pred = pred_r
                best_pred_idx = pi

        if best_pred is None or best_cover < config.recognition_cover_threshold:
            continue

        label = gt_r.label
        if label == "text":
            score = substring_cer(gt_r.transcription, best_pred.transcription,
                                  case_sensitive=config.text_case_sensitive)
            coverage_rec[label]["scores"].append(score)
        elif label == "liding":
            result = liding_score(gt_r.transcription, best_pred.transcription, config)
            coverage_rec[label]["scores"].append({
                "string_ned": result["string_ned"],
                "tree_ted": result.get("tree_ted"),
                "combined": result["combined"],
                "via_coverage": True,
            })
            score = result["combined"]

        if debug_pairs is not None:
            debug_pairs.append({
                "type": "coverage",
                "pred_idx": best_pred_idx,
                "gt_idx": j,
                "pred_label": best_pred.label,
                "gt_label": gt_r.label,
                "pred_bbox": best_pred.bbox,
                "gt_bbox": gt_r.bbox,
                "pred_text": best_pred.transcription[:80],
                "gt_text": gt_r.transcription[:80],
                "coverage": round(best_cover, 4),
                "rec_score": round(score, 4) if isinstance(score, (int, float)) else score,
            })

    # Merge coverage scores into recognition — all GT text regions should
    # have their OCR evaluated, even those that failed bbox matching.
    for label in TEXTUAL_LABELS:
        cr = coverage_rec[label]["scores"]
        if cr:
            recognition[label]["scores"].extend(cr)
            all_scores = recognition[label]["scores"]
            if label == "text":
                recognition[label]["mean"] = sum(all_scores) / len(all_scores)
            else:
                recognition[label]["mean"] = sum(
                    r["combined"] if isinstance(r, dict) else r for r in all_scores
                ) / len(all_scores)
            recognition[label]["num_pairs"] = len(all_scores)
            recognition[label]["num_coverage"] = len(cr)
            recognition_scores[label] = recognition[label]["mean"]

    # ── Composite ──
    composite = composite_score(
        detection_f1_by_label=detection_f1,
        recognition_score_by_label=recognition_scores,
        classification_acc=classification["accuracy"],
        config=config,
    )

    result = {
        "image_id": gt.image_id,
        "detection": detection,
        "classification": classification,
        "recognition": recognition,
        "composite": composite,
    }

    if debug_pairs is not None:
        # Track which GTs were evaluated (matched + coverage)
        evaluated_gt = matched_gt.copy()
        for p in debug_pairs:
            if p["type"] == "coverage":
                evaluated_gt.add(p["gt_idx"])

        # Collect unmatched predictions and GTs for debug display
        unmatched_preds = []
        for i, r in enumerate(pred.regions):
            if i not in matched_pred:
                unmatched_preds.append({
                    "idx": i, "label": r.label,
                    "bbox": r.bbox, "text": r.transcription[:80],
                })
        unmatched_gts = []
        for j, r in enumerate(gt.regions):
            if j not in evaluated_gt:
                unmatched_gts.append({
                    "idx": j, "label": r.label,
                    "bbox": r.bbox, "text": r.transcription[:80],
                })

        result["debug"] = {
            "pairs": debug_pairs,
            "unmatched_pred": unmatched_preds,
            "unmatched_gt": unmatched_gts,
            "num_pred": len(pred.regions),
            "num_gt": len(gt.regions),
        }

    return result


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