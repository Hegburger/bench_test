"""Evaluation metrics for document parsing benchmark."""

from benchmark.config import EvalConfig
from benchmark.ids_parser import parse_ids, ids_string_normalized
from benchmark.schema import Region


# ─── Text recognition ────────────────────────────────────────────────

def char_error_rate(reference: str, hypothesis: str, case_sensitive: bool = False) -> float:
    """Character Error Rate via Levenshtein edit distance.

    Returns a value in [0, 1] (higher is worse). 0 means perfect match.
    """
    if not case_sensitive:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    n = len(reference)
    m = len(hypothesis)

    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return 1.0
    if m == 0:
        return 1.0

    # DP with two rows
    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(
                    prev[j],       # delete
                    curr[j - 1],   # insert
                    prev[j - 1],   # substitute
                )
        prev, curr = curr, prev

    return prev[m] / max(n, m)


def substring_cer(reference: str, hypothesis: str, case_sensitive: bool = False) -> float:
    """Best-case CER by sliding a window of len(reference) over hypothesis.

    For granularity mismatch: when a layout-level prediction covers multiple
    GT regions, the GT text may be a substring of the long prediction text.
    This finds the best alignment before computing CER.

    Returns a value in [0, 1] (lower is better).
    Falls back to regular CER if hypothesis is not longer than reference.
    """
    n = len(reference)
    m = len(hypothesis)

    if n == 0 or m == 0:
        return char_error_rate(reference, hypothesis, case_sensitive)

    if m <= n:
        return char_error_rate(reference, hypothesis, case_sensitive)

    # Slide window, find min CER
    best = 1.0
    for start in range(m - n + 1):
        window = hypothesis[start:start + n]
        cer = char_error_rate(reference, window, case_sensitive)
        if cer < best:
            best = cer
        if best == 0.0:
            break

    return best


def normalized_edit_distance(reference: str, hypothesis: str, case_sensitive: bool = False) -> float:
    """Alias for CER, synonym."""
    return char_error_rate(reference, hypothesis, case_sensitive)


# ─── Liding recognition ──────────────────────────────────────────────

def ids_tree_distance(reference: str, hypothesis: str, config: EvalConfig) -> float | None:
    """Compute normalized tree edit distance between two IDS strings.

    Returns None if either string cannot be parsed as IDS.
    Returns a value in [0, 1] (higher is worse).
    """
    ref_norm = ids_string_normalized(reference)
    hyp_norm = ids_string_normalized(hypothesis)

    tree_ref = parse_ids(ref_norm)
    tree_hyp = parse_ids(hyp_norm)

    if tree_ref is None or tree_hyp is None:
        return None

    from benchmark.tree_edit import normalized_tree_edit_distance
    return normalized_tree_edit_distance(
        tree_ref, tree_hyp,
        operators=config.ids_operators,
        cost_same=config.cost_rename_identical,
        cost_both_op=config.cost_rename_both_operators,
        cost_diff=config.cost_rename_otherwise,
        cost_delete=config.cost_delete,
        cost_insert=config.cost_insert,
    )


def liding_score(reference: str, hypothesis: str, config: EvalConfig) -> dict:
    """Compute liding recognition score combining string and tree metrics.

    Returns a dict with keys: 'string_ned', 'tree_ted', 'combined'.
    combined = (string_ned + tree_ted) / 2 (or just string_ned if tree fails)
    """
    string_ned = normalized_edit_distance(reference, hypothesis)

    result = {"string_ned": string_ned}

    if config.liding_use_tree_edit:
        tree_ted = ids_tree_distance(reference, hypothesis, config)
        result["tree_ted"] = tree_ted
        if tree_ted is not None:
            result["combined"] = (string_ned + tree_ted) / 2.0
        else:
            result["combined"] = string_ned
    else:
        result["combined"] = string_ned

    return result


# ─── Detection metrics ───────────────────────────────────────────────

def detection_metrics(
    num_pred: int, num_gt: int, num_matched: int,
) -> dict:
    """Compute precision, recall, F1 for detection.

    When both num_pred and num_gt are zero for a label, the image contains
    no instance of that label and the model makes no prediction — this is
    perfect agreement, so F1 = 1.0.
    """
    if num_pred == 0 and num_gt == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "num_pred": 0,
            "num_gt": 0,
            "num_matched": 0,
        }
    precision = num_matched / num_pred if num_pred > 0 else 0.0
    recall = num_matched / num_gt if num_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_pred": num_pred,
        "num_gt": num_gt,
        "num_matched": num_matched,
    }


# ─── Classification metrics ──────────────────────────────────────────

def classification_accuracy(
    matched_pairs: list[tuple[Region, Region]],
) -> dict:
    """Compute label classification accuracy for matched pairs."""
    if not matched_pairs:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "confusion": {}}

    correct = 0
    confusion = {}  # "pred_label -> gt_label" -> count

    for pred, gt in matched_pairs:
        key = f"{pred.label} -> {gt.label}"
        confusion[key] = confusion.get(key, 0) + 1
        if pred.label == gt.label:
            correct += 1

    return {
        "accuracy": correct / len(matched_pairs),
        "correct": correct,
        "total": len(matched_pairs),
        "confusion": confusion,
    }


# ─── Composite ───────────────────────────────────────────────────────

def composite_score(
    detection_f1_by_label: dict[str, float],
    recognition_score_by_label: dict[str, float],
    classification_acc: float,
    config: EvalConfig,
) -> float:
    """Weighted composite score across all label types.

    For each label:
      label_score = detection_f1 * (recognition_score if textual else 1.0)

    Composite = weighted sum of label scores, penalized by (1 - class_acc).
    """
    label_scores = {}

    # Text
    text_det = detection_f1_by_label.get("text", 0.0)
    text_rec = recognition_score_by_label.get("text", 0.0)
    label_scores["text"] = text_det * (1.0 - text_rec)  # higher rec_score = worse, so invert

    # Liding
    liding_det = detection_f1_by_label.get("liding", 0.0)
    liding_rec = recognition_score_by_label.get("liding", 0.0)
    label_scores["liding"] = liding_det * (1.0 - liding_rec)

    # Oracle (detection only)
    oracle_det = detection_f1_by_label.get("oracle", 0.0)
    label_scores["oracle"] = oracle_det

    # Picture (detection only)
    picture_det = detection_f1_by_label.get("picture", 0.0)
    label_scores["picture"] = picture_det

    composite = (
        config.weight_text * label_scores["text"]
        + config.weight_liding * label_scores["liding"]
        + config.weight_oracle * label_scores["oracle"]
        + config.weight_picture * label_scores["picture"]
    )

    # Penalize classification errors
    composite *= classification_acc

    # Also factor in unmapped labels if configured
    return composite