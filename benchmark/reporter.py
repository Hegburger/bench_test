"""Report generation: text console output and JSON export."""

import json
from pathlib import Path


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
