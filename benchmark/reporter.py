"""Report generation: text console output and JSON export."""

import json
from pathlib import Path


def _bbox_str(bbox: list[float]) -> str:
    return f"[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"


def _ok_fail(score: float, threshold: float = 0.3) -> str:
    if score <= threshold:
        return "OK"
    return "FAIL"


def print_summary(results: dict) -> None:
    """Print a human-readable summary to console."""
    summary = results.get("summary", {})
    config = results.get("config", {})

    print("=" * 64)
    print("  DOCUMENT PARSING BENCHMARK — RESULTS")
    print("=" * 64)

    print(f"\n  Images evaluated: {summary.get('num_evaluated', 0)} / {summary.get('num_images', 0)}")
    print(f"  IOU threshold: {config.get('iou_threshold', 'N/A')}")
    print(f"  Tree edit distance: {'enabled' if config.get('liding_use_tree_edit') else 'disabled'}")

    # Detection
    det = summary.get("detection", {})
    print("\n  ── DETECTION ──")
    print(f"  {'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for label in ("text", "liding", "oracle", "picture", "overall"):
        d = det.get(label, {})
        print(f"  {label:<12} {d.get('mean_precision', 0):>10.3f} {d.get('mean_recall', 0):>10.3f} {d.get('mean_f1', 0):>10.3f}")

    # Recognition
    rec = summary.get("recognition", {})
    print("\n  ── RECOGNITION (error rate, lower is better) ──")
    for label, info in rec.items():
        print(f"  {label:<12} mean_error: {info.get('mean_error_rate', 0):.4f}")

    # Classification
    print(f"\n  ── CLASSIFICATION ──")
    print(f"  Accuracy: {summary.get('classification_accuracy', 0):.4f}")

    # Composite
    print(f"\n  ── COMPOSITE ──")
    print(f"  Score: {summary.get('mean_composite', 0):.4f}")

    print("\n" + "=" * 64)


def save_json_report(results: dict, output_path: str | Path) -> None:
    """Save full evaluation results as JSON."""
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def compare_models(
    model_results: dict[str, dict],
    output_path: str | Path | None = None,
) -> None:
    """Print comparison table for multiple models.

    Args:
        model_results: {model_name: full_evaluation_results}
    """
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON")
    print("=" * 72)

    header = f"  {'Model':<20} {'Det-F1':>8} {'Cls-Acc':>8} {'Text-Err':>8} {'Liding-Err':>8} {'Composite':>10}"
    print(header)
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    rows = []
    for name, results in model_results.items():
        s = results.get("summary", {})
        det = s.get("detection", {}).get("overall", {})
        rec = s.get("recognition", {})

        det_f1 = det.get("mean_f1", 0)
        cls_acc = s.get("classification_accuracy", 0)
        text_err = rec.get("text", {}).get("mean_error_rate", 0)
        liding_err = rec.get("liding", {}).get("mean_error_rate", 0)
        composite = s.get("mean_composite", 0)

        print(f"  {name:<20} {det_f1:>8.3f} {cls_acc:>8.3f} {text_err:>8.4f} {liding_err:>8.4f} {composite:>10.4f}")

        rows.append({
            "model": name,
            "detection_f1": det_f1,
            "classification_accuracy": cls_acc,
            "text_error_rate": text_err,
            "liding_error_rate": liding_err,
            "composite": composite,
        })

    print("=" * 72 + "\n")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


def print_debug_report(model_name: str, per_image: dict) -> None:
    """Print detailed per-image matching and recognition report for debugging.

    Shows every prediction, which GT it matched to, the match method (IOU vs
    containment vs coverage), and the recognition score for each pair.
    """
    W = 80

    for image_id, result in per_image.items():
        if "error" in result:
            print(f"\n{'='*W}")
            print(f"  {image_id}  —  ERROR: {result['error']}")
            print(f"{'='*W}")
            continue

        debug = result.get("debug", {})
        if not debug:
            print(f"\n  {image_id}: no debug data (run with --debug)")
            continue

        pairs = debug["pairs"]
        unmatched_pred = debug["unmatched_pred"]
        unmatched_gt = debug["unmatched_gt"]

        print(f"\n{'='*W}")
        print(f"  {image_id}  —  {debug['num_pred']} predictions  vs  {debug['num_gt']} GT regions")
        print(f"{'='*W}")

        # ── Matched pairs ──
        matched = [p for p in pairs if p["type"] == "matched"]
        if matched:
            print(f"\n  [MATCHED PAIRS]  ({len(matched)} pairs)")
            print(f"  {'':>4} {'Method':<12} {'Pred':<6} {'GT':<6} {'Rec':>6}  {'Status':<6}")
            print(f"  {'':>4} {'-'*12} {'-'*6} {'-'*6} {'-'*6}  {'-'*6}")
            for p in matched:
                method = p.get("match_method", "?")
                score = p.get("match_score", 0)
                rec = p.get("rec_score")
                pred_label = p["pred_label"]
                gt_label = p["gt_label"]
                pred_text = p.get("pred_text", "")[:50]
                gt_text = p.get("gt_text", "")[:50]

                if rec is not None:
                    status = _ok_fail(rec)
                    rec_str = f"{rec:.4f}"
                else:
                    status = "SKIP"
                    rec_str = "N/A"

                print(f"  P[{p['pred_idx']:>2}] {method:<12} {pred_label:<6} → GT[{p['gt_idx']:>2}] {gt_label:<6} {rec_str:>6}  {status:<6}")
                print(f"       pred: {_bbox_str(p['pred_bbox'])}  \"{pred_text}\"")
                print(f"       gt:   {_bbox_str(p['gt_bbox'])}  \"{gt_text}\"")

        # ── Coverage pairs ──
        coverage = [p for p in pairs if p["type"] == "coverage"]
        if coverage:
            print(f"\n  [COVERAGE RECOGNITION]  ({len(coverage)} pairs)")
            print(f"  {'':>4} {'Cover':>8}  {'GT':<6} {'Rec':>6}  {'Status':<6}")
            print(f"  {'':>4} {'-'*8}  {'-'*6} {'-'*6}  {'-'*6}")
            for p in coverage:
                cov = p.get("coverage", 0)
                rec = p.get("rec_score", 1.0)
                gt_label = p["gt_label"]
                gt_text = p.get("gt_text", "")[:50]
                pred_text = p.get("pred_text", "")[:50]
                status = _ok_fail(rec)

                print(f"       {cov:>8.3f}  GT[{p['gt_idx']:>2}] {gt_label:<6} {rec:.4f}  {status:<6}")
                print(f"       ← P[{p['pred_idx']:>2}] {p['pred_label']:<6}  \"{gt_text}\"")
                print(f"         pred: \"{pred_text}\"")

        # ── Unmatched predictions ──
        if unmatched_pred:
            print(f"\n  [UNMATCHED PREDICTIONS]  ({len(unmatched_pred)})")
            for up in unmatched_pred:
                print(f"  P[{up['idx']:>2}] {up['label']:<10} {_bbox_str(up['bbox'])}  \"{up['text']}\"")

        # ── Unmatched GT ──
        if unmatched_gt:
            print(f"\n  [UNMATCHED GT]  ({len(unmatched_gt)})")
            for ug in unmatched_gt:
                print(f"  GT[{ug['idx']:>2}] {ug['label']:<10} {_bbox_str(ug['bbox'])}  \"{ug['text']}\"")
            # Group by label for summary
            from collections import Counter
            label_counts = Counter(ug["label"] for ug in unmatched_gt)
            parts = ", ".join(f"{c}x {l}" for l, c in sorted(label_counts.items()))
            print(f"       → {parts}")

        # ── Recognition summary per label ──
        rec = result.get("recognition", {})
        print(f"\n  [RECOGNITION SUMMARY]")
        for label in ("text", "liding"):
            rd = rec.get(label, {})
            direct = rd.get("num_pairs", 0) - rd.get("num_coverage", 0)
            cov = rd.get("num_coverage", 0)
            mean = rd.get("mean", 0)
            print(f"  {label:<10} direct={direct}, coverage={cov}, mean_err={mean:.4f}")

        # Classification
        cls_info = result.get("classification", {})
        if cls_info.get("confusion"):
            print(f"\n  [CLASSIFICATION]  acc={cls_info['accuracy']:.3f}")
            for k, v in cls_info["confusion"].items():
                print(f"  {k}: {v}")

        print(f"\n  Composite: {result['composite']:.4f}")
