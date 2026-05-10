"""Benchmark for ancient document parsing evaluation."""

from benchmark.config import EvalConfig, LABEL_TYPES, TEXTUAL_LABELS, DETECTION_ONLY_LABELS
from benchmark.schema import Region, ImageAnnotation
from benchmark.data_loader import load_gt_file, load_gt_directory
from benchmark.evaluator import evaluate_image, evaluate_dataset
from benchmark.reporter import print_summary, save_json_report, compare_models, print_debug_report
from benchmark.ids_parser import parse_ids, IDSNode
from benchmark.tree_edit import tree_edit_distance, normalized_tree_edit_distance
from benchmark.matcher import compute_iou, match_regions
from benchmark.metrics import char_error_rate, normalized_edit_distance, liding_score

__all__ = [
    "EvalConfig",
    "LABEL_TYPES",
    "TEXTUAL_LABELS",
    "DETECTION_ONLY_LABELS",
    "Region",
    "ImageAnnotation",
    "load_gt_file",
    "load_gt_directory",
    "evaluate_image",
    "evaluate_dataset",
    "print_summary",
    "save_json_report",
    "compare_models",
    "print_debug_report",
    "parse_ids",
    "IDSNode",
    "tree_edit_distance",
    "normalized_tree_edit_distance",
    "compute_iou",
    "match_regions",
    "char_error_rate",
    "normalized_edit_distance",
    "liding_score",
]