"""Evaluation configuration."""

from dataclasses import dataclass, field

# Valid label types
LABEL_TYPES = ("text", "liding", "oracle", "picture")

# Region labels that carry text content for recognition evaluation
TEXTUAL_LABELS = ("text", "liding")

# Region labels that are evaluated by detection only (IOU)
DETECTION_ONLY_LABELS = ("oracle", "picture")


@dataclass
class EvalConfig:
    """Configuration for benchmark evaluation."""

    # --- Matching ---
    iou_threshold: float = 0.5
    """Minimum IOU for a predicted region to be considered a match."""

    containment_threshold: float = 0.7
    """Minimum containment ratio (max(gt_in_pred, pred_in_gt)) for matching
    when IOU fails. Handles granularity mismatch between layout-level predictions
    and character-level ground truth."""

    recognition_cover_threshold: float = 0.5
    """Minimum coverage ratio for a GT text region to be evaluated for
    recognition against an unmatched prediction via substring CER."""

    # --- Debug ---
    debug: bool = False
    """If True, include per-pair matching and recognition details in results."""

    # --- Text metrics ---
    text_case_sensitive: bool = False
    """Whether text CER is case-sensitive."""

    # --- Liding metrics ---
    liding_use_tree_edit: bool = True
    """If True, compute tree edit distance for IDS strings in addition to string NED."""

    # --- composite weights ---
    weight_text: float = 0.35
    """Weight of text regions in composite score."""

    weight_liding: float = 0.40
    """Weight of liding regions in composite score."""

    weight_oracle: float = 0.15
    """Weight of oracle regions in composite score."""

    weight_picture: float = 0.10
    """Weight of picture regions in composite score."""

    # --- IDS operators ---
    ids_operators: set[str] = field(default_factory=lambda: {
        "⿰", "⿱", "⿲", "⿳", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻",
    })
    """Unicode range U+2FF0-U+2FFF — IDS operator characters."""

    # --- Tree edit cost ---
    cost_rename_identical: float = 0.0
    cost_rename_both_operators: float = 0.5
    cost_rename_otherwise: float = 1.0
    cost_delete: float = 1.0
    cost_insert: float = 1.0
