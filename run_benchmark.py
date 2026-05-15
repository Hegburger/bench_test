"""CLI entry point for the document parsing benchmark.

Usage:
    python run_benchmark.py                          # evaluate all GT with all adapters
    python run_benchmark.py --gt-dir data/GT         # specify GT directory
    python run_benchmark.py --output results.json    # specify output file
    python run_benchmark.py --iou 0.6                # customize IOU threshold
"""

import argparse
import json
import sys
from pathlib import Path

# Fix Unicode output on Windows GBK terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from benchmark import (
    EvalConfig,
    load_gt_directory,
    evaluate_dataset,
    print_summary,
    save_json_report,
    compare_models,
    print_debug_report,
)
from benchmark.adapters import PaddleOCRAdapter, PaddleOCRVAdapter, GeminiAdapter


def create_dummy_predictions(
    ground_truths: dict,
    noise_bbox: float = 2.0,
    noise_text: float = 0.1,
) -> dict:
    """Create synthetic predictions from GT for testing the pipeline.

    Adds small perturbations to bbox and text to simulate model noise.
    """
    import random

    predictions = {}
    for image_id, gt_ann in ground_truths.items():
        regions = []
        for r in gt_ann.regions:
            # Perturb bbox (small noise, regions can be tiny ~4x2 px)
            bbox = list(r.bbox)
            for i in range(4):
                bbox[i] += random.uniform(-noise_bbox, noise_bbox)

            # Perturb text (swap random characters)
            text = r.transcription
            if text and random.random() < noise_text:
                chars = list(text)
                for _ in range(max(1, len(chars) // 15)):
                    idx = random.randint(0, len(chars) - 1)
                    chars[idx] = chr(ord(chars[idx]) + random.randint(-1, 1))
                text = "".join(chars)

            regions.append(type(r)(
                bbox=bbox,
                label=r.label,
                transcription=text,
                confidence=random.uniform(0.7, 0.99),
            ))

        # Drop ~10% of regions (at least 1 if > 5 regions)
        if len(regions) > 10:
            n_drop = max(1, len(regions) // 10)
            for _ in range(n_drop):
                regions.pop(random.randint(0, len(regions) - 1))
        elif len(regions) >= 4:
            # For small images, maybe drop 1
            if random.random() < 0.3:
                regions.pop(random.randint(0, len(regions) - 1))

        from benchmark.schema import ImageAnnotation
        pred = ImageAnnotation(
            image_id=image_id,
            image_path=gt_ann.image_path,
            regions=regions,
        )
        predictions[image_id] = pred

    return predictions


def load_model_predictions(
    model_dir: Path,
    ground_truths: dict,
    adapter,
) -> dict:
    """Load model predictions from JSON files using an adapter.

    Expects structure: model_dir/*.json where each JSON contains the
    model's raw output for one image. Filenames should match image_ids.
    """
    predictions = {}
    for image_id, gt_ann in ground_truths.items():
        # Try exact match first
        json_path = model_dir / f"{image_id}.json"
        if not json_path.exists():
            # Try with _pred suffix
            json_path = model_dir / f"{image_id}_pred.json"
        if not json_path.exists():
            # Try matching any file containing the image_id
            candidates = list(model_dir.glob(f"*{image_id}*.json"))
            if candidates:
                json_path = candidates[0]
            else:
                continue

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        pred = adapter.parse(raw, image_id=image_id, image_path=gt_ann.image_path)
        predictions[image_id] = pred

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Ancient Document Parsing Benchmark")
    parser.add_argument("--gt-dir", default="data/GT", help="Directory containing GT JSON files")
    parser.add_argument("--output", default=None, help="Output JSON path for full results")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold for matching")
    parser.add_argument("--no-tree-edit", action="store_true", help="Disable tree edit distance for liding")
    parser.add_argument("--synthetic", action="store_true", default=True,
                       help="Use synthetic (noised GT) predictions for testing")
    parser.add_argument("--model-dir", default=None,
                       help="Directory with model prediction JSON files")
    parser.add_argument("--model", default=None,
                       help="Model adapter to use: paddle_ocr, paddle_ocr_vl, gemini")
    parser.add_argument("--debug", action="store_true",
                       help="Print detailed per-pair matching and recognition report")

    args = parser.parse_args()

    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    gt_dir = script_dir / args.gt_dir
    if not gt_dir.exists():
        print(f"Error: GT directory not found: {gt_dir}")
        sys.exit(1)

    # Load GT
    print(f"Loading ground truth from {gt_dir}...")
    ground_truths_list = load_gt_directory(gt_dir)
    ground_truths = {ann.image_id: ann for ann in ground_truths_list}
    print(f"  Loaded {len(ground_truths)} annotated images")

    if not ground_truths:
        print("Error: No GT annotations found")
        sys.exit(1)

    # Config
    config = EvalConfig(
        iou_threshold=args.iou,
        liding_use_tree_edit=not args.no_tree_edit,
        debug=args.debug,
    )

    # Models to evaluate
    model_results = {}

    if args.model_dir:
        # Use actual model predictions
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            print(f"Error: model directory not found: {model_dir}")
            sys.exit(1)

        adapter_map = {
            "paddle_ocr": PaddleOCRAdapter,
            "paddle_ocr_vl": PaddleOCRVAdapter,
            "gemini": GeminiAdapter,
        }

        for name, adapter_cls in adapter_map.items():
            if args.model and args.model != name:
                continue
            adapter = adapter_cls()
            predictions = load_model_predictions(model_dir, ground_truths, adapter)
            if predictions:
                print(f"\nEvaluating {adapter.model_name}...")
                results = evaluate_dataset(predictions, ground_truths, config)
                model_results[adapter.model_name] = results
                print_summary(results)
                if args.debug:
                    per_image = results.get("per_image", {})
                    print_debug_report(adapter.model_name, per_image)
    else:
        # Synthetic test
        print("\nUsing synthetic (noised GT) predictions for pipeline testing...")
        predictions = create_dummy_predictions(ground_truths)
        results = evaluate_dataset(predictions, ground_truths, config)
        model_results["SyntheticTest"] = results
        print_summary(results)

    # Compare if multiple models
    if len(model_results) > 1:
        compare_models(model_results)

    # Save
    if args.output:
        output_path = script_dir / args.output
    else:
        output_path = script_dir / "benchmark_results.json"

    # Save all model results
    all_results = {
        "models": {},
        "gt_dir": str(gt_dir),
        "config": {
            "iou_threshold": config.iou_threshold,
            "liding_use_tree_edit": config.liding_use_tree_edit,
        },
    }
    for name, r in model_results.items():
        all_results["models"][name] = r

    save_json_report(all_results, output_path)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
