"""Zhang-Shasha tree edit distance for IDS parse trees.

Computes the minimum-cost sequence of edit operations (rename, delete, insert)
to transform one ordered tree into another.

Reference: Zhang & Shasha (1989), "Simple Fast Algorithms for the Editing
Distance Between Trees and Related Problems."
"""

from benchmark.ids_parser import IDSNode


def _cost_rename(a: str, b: str, operators: set[str],
                 cost_same: float, cost_both_op: float, cost_diff: float) -> float:
    if a == b:
        return cost_same
    if a in operators and b in operators:
        return cost_both_op
    return cost_diff


def tree_edit_distance(
    tree_a: IDSNode,
    tree_b: IDSNode,
    operators: set[str],
    cost_same: float = 0.0,
    cost_both_op: float = 0.5,
    cost_diff: float = 1.0,
    cost_delete: float = 1.0,
    cost_insert: float = 1.0,
) -> float:
    """Compute tree edit distance between two IDS trees.

    Returns the raw distance (not normalized).
    """
    # Get postorder traversals
    post_a = tree_a.postorder()
    post_b = tree_b.postorder()

    n = len(post_a)
    m = len(post_b)

    # Map postorder index -> node for leftmost leaf computation
    nodes_a = _postorder_nodes(tree_a)
    nodes_b = _postorder_nodes(tree_b)

    # Build leftmost leaf index for each node
    lm_a = _leftmost_leaf(nodes_a)
    lm_b = _leftmost_leaf(nodes_b)

    # DP table: treedist[i][j] = distance between subtrees rooted at
    # postorder index i in A and postorder index j in B
    # We'll use the efficient tree distance algorithm.
    # For small trees, a simpler O(n^2 * m^2) forest distance is acceptable.

    # Forest distance DP: fd[i][j] for subforests
    # Use Zhang-Shasha's decomposition

    # Actually, for clarity, use a dictionary-based forest distance
    treedist = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            # Roots of the two subtrees
            ni = nodes_a[i]
            nj = nodes_b[j]

            # Compute forest distance between the two subtrees
            treedist[i][j] = _forest_distance(
                ni, nj, lm_a, lm_b, operators,
                cost_same, cost_both_op, cost_diff,
                cost_delete, cost_insert,
                nodes_a, nodes_b,
            )

    return treedist[n - 1][m - 1]


def _forest_distance(
    node_i: IDSNode,
    node_j: IDSNode,
    lm_a: dict[int, int],
    lm_b: dict[int, int],
    operators: set[str],
    cost_same: float,
    cost_both_op: float,
    cost_diff: float,
    cost_delete: float,
    cost_insert: float,
    nodes_a: list[IDSNode],
    nodes_b: list[IDSNode],
) -> float:
    """Compute forest distance between subtrees rooted at node_i and node_j."""

    # Get the subtree nodes in postorder
    sub_a = _subtree_postorder(node_i, nodes_a)
    sub_b = _subtree_postorder(node_j, nodes_b)

    len_a = len(sub_a)
    len_b = len(sub_b)

    # fd[p][q] = forest distance for first p nodes of sub_a and first q nodes of sub_b
    fd = [[0.0] * (len_b + 1) for _ in range(len_a + 1)]

    # Initialize delete costs
    for p in range(1, len_a + 1):
        idx_a = sub_a[p - 1]
        fd[p][0] = fd[p - 1][0] + cost_delete

    # Initialize insert costs
    for q in range(1, len_b + 1):
        fd[0][q] = fd[0][q - 1] + cost_insert

    # Find postorder index -> node mapping
    node_a_map = {id(n): i for i, n in enumerate(nodes_a)}
    node_b_map = {id(n): i for i, n in enumerate(nodes_b)}

    for p in range(1, len_a + 1):
        for q in range(1, len_b + 1):
            idx_a = sub_a[p - 1]
            idx_b = sub_b[q - 1]

            na = nodes_a[idx_a]
            nb = nodes_b[idx_b]

            # Option 1: delete node na
            fd[p][q] = fd[p - 1][q] + cost_delete

            # Option 2: insert node nb
            if fd[p][q - 1] + cost_insert < fd[p][q]:
                fd[p][q] = fd[p][q - 1] + cost_insert

            # Option 3: rename na -> nb (if both are roots of their respective
            # subtrees within the current forest decomposition)
            # Check if leftmost leaf of na is the first node in sub_a[p-1..]
            # and leftmost leaf of nb is the first node in sub_b[q-1..]
            lma = lm_a.get(idx_a, idx_a)
            lmb = lm_b.get(idx_b, idx_b)

            # In postorder, the subtree rooted at na spans from lma to idx_a
            # and the subtree rooted at nb spans from lmb to idx_b
            # If lma and lmb are the "first" nodes in the current forest
            # (i.e., they match the start of the subforests), we can rename
            # and recurse on the remaining forest
            if lma == sub_a[0] and lmb == sub_b[0]:
                # p_len = size of subtree rooted at na = idx_a - lma + 1
                subtree_size_a = idx_a - lma + 1
                subtree_size_b = idx_b - lmb + 1

                rename_cost = _cost_rename(
                    na.value, nb.value, operators,
                    cost_same, cost_both_op, cost_diff,
                )

                # fd at the left subtrees + rename cost
                candidate = (
                    fd[p - subtree_size_a][q - subtree_size_b]
                    + rename_cost
                )
                if candidate < fd[p][q]:
                    fd[p][q] = candidate

    return fd[len_a][len_b]


def _postorder_nodes(root: IDSNode) -> list[IDSNode]:
    """Return all nodes in postorder as a list."""
    result = []

    def dfs(node: IDSNode):
        for child in node.children:
            dfs(child)
        result.append(node)

    dfs(root)
    return result


def _subtree_postorder(root: IDSNode, all_nodes: list[IDSNode]) -> list[int]:
    """Return indices (into all_nodes) of the subtree rooted at root, in postorder."""
    # Find the root index
    root_idx = None
    for i, n in enumerate(all_nodes):
        if n is root:
            root_idx = i
            break
    if root_idx is None:
        return []

    # Find the leftmost leaf index
    lm = _find_leftmost(root, all_nodes)
    return list(range(lm, root_idx + 1))


def _leftmost_leaf(nodes: list[IDSNode]) -> dict[int, int]:
    """Map node postorder index -> postorder index of its leftmost leaf."""
    # For each node, find leftmost leaf by tracking first child chain
    result = {}
    for i, node in enumerate(nodes):
        result[i] = _find_leftmost(node, nodes)
    return result


def _find_leftmost(root: IDSNode, all_nodes: list[IDSNode]) -> int:
    """Find the postorder index of the leftmost leaf descendant of root."""
    # Walk down the leftmost child chain
    node = root
    while node.children:
        node = node.children[0]
    # Find this node in all_nodes
    for i, n in enumerate(all_nodes):
        if n is node:
            return i
    return 0


def normalized_tree_edit_distance(
    tree_a: IDSNode,
    tree_b: IDSNode,
    operators: set[str],
    **cost_kwargs,
) -> float:
    """Compute normalized tree edit distance in [0, 1].

    Normalized by the sum of tree sizes (number of nodes).
    """
    raw = tree_edit_distance(tree_a, tree_b, operators, **cost_kwargs)
    size_sum = len(tree_a.postorder()) + len(tree_b.postorder())
    if size_sum == 0:
        return 0.0
    return raw / size_sum