"""Region matching via IOU + Hungarian algorithm.

Matches predicted regions to ground-truth regions using 1-to-1 bipartite
matching that maximizes total IOU, subject to a minimum IOU threshold.
"""

import itertools

from benchmark.schema import Region


def compute_iou(a: Region, b: Region) -> float:
    """Compute Intersection-over-Union of two regions' bounding boxes."""
    xa1, ya1, xa2, ya2 = a.bbox
    xb1, yb1, xb2, yb2 = b.bbox

    # Intersection
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)

    if xi1 >= xi2 or yi1 >= yi2:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    union_area = a.area + b.area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def hungarian_match(
    cost_matrix: list[list[float]],
) -> list[tuple[int, int]]:
    """Solve assignment problem: minimize total cost.

    Args:
        cost_matrix: cost_matrix[i][j] = cost of matching pred i to gt j.

    Returns:
        List of (pred_idx, gt_idx) pairs.

    For small matrices, uses exhaustive search over the smaller dimension.
    For larger matrices (max_dim > 8), uses a greedy fallback.
    """
    n_pred = len(cost_matrix)
    n_gt = len(cost_matrix[0]) if cost_matrix else 0

    if n_pred == 0 or n_gt == 0:
        return []

    # For small matrices, brute-force is reliable
    small_dim = min(n_pred, n_gt)
    large_dim = max(n_pred, n_gt)

    if small_dim <= 8:
        return _hungarian_bruteforce(cost_matrix, n_pred, n_gt)
    else:
        return _hungarian_greedy(cost_matrix, n_pred, n_gt)


def _hungarian_bruteforce(
    cost_matrix: list[list[float]],
    n_pred: int,
    n_gt: int,
) -> list[tuple[int, int]]:
    """Exhaustive search over assignments."""
    small_is_pred = n_pred <= n_gt
    small_n = min(n_pred, n_gt)
    large_n = max(n_pred, n_gt)

    best_cost = float("inf")
    best_assignment = []

    for perm in itertools.permutations(range(large_n), small_n):
        total = 0.0
        for i, j in enumerate(perm):
            if small_is_pred:
                total += cost_matrix[i][j]
            else:
                total += cost_matrix[j][i]
        if total < best_cost:
            best_cost = total
            if small_is_pred:
                best_assignment = [(i, j) for i, j in enumerate(perm)]
            else:
                best_assignment = [(j, i) for i, j in enumerate(perm)]

    return best_assignment


def _hungarian_greedy(
    cost_matrix: list[list[float]],
    n_pred: int,
    n_gt: int,
) -> list[tuple[int, int]]:
    """Greedy matching: sort by cost, take cheapest available pairs."""
    all_pairs = []
    for i in range(n_pred):
        for j in range(n_gt):
            all_pairs.append((cost_matrix[i][j], i, j))
    all_pairs.sort()

    used_pred = set()
    used_gt = set()
    result = []

    for cost, i, j in all_pairs:
        if i not in used_pred and j not in used_gt:
            result.append((i, j))
            used_pred.add(i)
            used_gt.add(j)

    return result


def _matching_cost(
    pred: Region,
    gt: Region,
    iou_threshold: float,
    center_dist_threshold: float,
) -> float:
    """Compute matching cost between two regions.

    Returns cost in [0, 1] for matchable pairs, or 1e9 for unmatchable.
    Uses IOU primarily, with center-distance fallback for small regions.
    """
    iou = compute_iou(pred, gt)

    if iou >= iou_threshold:
        return 1.0 - iou

    # For very small regions, check center distance as fallback
    pred_area = pred.area
    gt_area = gt.area
    if pred_area < 100 and gt_area < 100:
        pred_cx = (pred.bbox[0] + pred.bbox[2]) / 2
        pred_cy = (pred.bbox[1] + pred.bbox[3]) / 2
        gt_cx = (gt.bbox[0] + gt.bbox[2]) / 2
        gt_cy = (gt.bbox[1] + gt.bbox[3]) / 2
        dist = ((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2) ** 0.5
        avg_size = (pred_area ** 0.5 + gt_area ** 0.5) / 2
        if dist < max(center_dist_threshold, avg_size * 2):
            return dist / center_dist_threshold

    return 1e9


def match_regions(
    pred_regions: list[Region],
    gt_regions: list[Region],
    iou_threshold: float = 0.5,
    center_dist_threshold: float = 20.0,
) -> dict:
    """Match predicted regions to GT regions via IOU + Hungarian.

    For extremely small regions (area < 100 px^2), also considers
    center-point distance as a matching criterion when IOU is low.

    Args:
        iou_threshold: Minimum IOU for a match.
        center_dist_threshold: Max center distance (pixels) for tiny region matching.

    Returns a dict with:
        matches: list of (pred_idx, gt_idx, iou) for matched pairs
        unmatched_pred: list of pred_idx
        unmatched_gt: list of gt_idx
        per_label: dict mapping label -> match details
    """
    if not pred_regions or not gt_regions:
        return {
            "matches": [],
            "unmatched_pred": list(range(len(pred_regions))),
            "unmatched_gt": list(range(len(gt_regions))),
            "per_label": {},
        }

    n_pred = len(pred_regions)
    n_gt = len(gt_regions)

    # Build cost matrix using helper
    cost_matrix = []
    for i in range(n_pred):
        row = []
        for j in range(n_gt):
            cost = _matching_cost(
                pred_regions[i], gt_regions[j],
                iou_threshold, center_dist_threshold,
            )
            row.append(cost)
        cost_matrix.append(row)

    # Hungarian matching
    assigned = hungarian_match(cost_matrix)

    # Filter out invalid matches (cost >= 1e9)
    matches = []
    matched_pred = set()
    matched_gt = set()

    for pred_idx, gt_idx in assigned:
        if cost_matrix[pred_idx][gt_idx] < 1e8:
            iou = 1.0 - cost_matrix[pred_idx][gt_idx]
            matches.append((pred_idx, gt_idx, iou))
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)

    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_gt = [j for j in range(n_gt) if j not in matched_gt]

    # Per-label breakdown
    per_label = {}
    for label in set(r.label for r in gt_regions) | set(r.label for r in pred_regions):
        pred_indices = [i for i, r in enumerate(pred_regions) if r.label == label]
        gt_indices = [j for j, r in enumerate(gt_regions) if r.label == label]

        # Sub-matrix for this label
        n_p = len(pred_indices)
        n_g = len(gt_indices)
        if n_p == 0 or n_g == 0:
            per_label[label] = {
                "matches": [],
                "num_pred": n_p,
                "num_gt": n_g,
                "precision": 0.0,
                "recall": 0.0,
            }
            continue

        sub_cost = []
        for pi in pred_indices:
            row = []
            for gj in gt_indices:
                cost = _matching_cost(
                    pred_regions[pi], gt_regions[gj],
                    iou_threshold, center_dist_threshold,
                )
                row.append(cost)
            sub_cost.append(row)

        sub_assigned = hungarian_match(sub_cost)
        label_matches = []
        for pi_sub, gj_sub in sub_assigned:
            if sub_cost[pi_sub][gj_sub] < 1e8:
                iou = 1.0 - sub_cost[pi_sub][gj_sub]
                label_matches.append((
                    pred_indices[pi_sub],
                    gt_indices[gj_sub],
                    iou,
                ))

        tp = len(label_matches)
        precision = tp / n_p if n_p > 0 else 0.0
        recall = tp / n_g if n_g > 0 else 0.0

        per_label[label] = {
            "matches": label_matches,
            "num_pred": n_p,
            "num_gt": n_g,
            "precision": precision,
            "recall": recall,
        }

    return {
        "matches": matches,
        "unmatched_pred": unmatched_pred,
        "unmatched_gt": unmatched_gt,
        "per_label": per_label,
    }