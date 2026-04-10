"""SSDBSCAN — semi-supervised density-based clustering by expansion.

This is Algorithm 1 from the SSDBCODI paper. We expand from each labeled normal
point using a Prim-like minimum-reachability traversal, and back-trace at the
maximum edge to assign cluster membership while respecting not-link constraints.

Key modification compared to the paper's pseudo-code: instead of a per-pair
distance lookup, we precompute the full pairwise distance matrix once using the
provided distance object's `pairwise()` method. This is dramatically faster for
the moderate dataset sizes the dashboard targets (< 10k points).
"""
import heapq
import numpy as np
from typing import Dict, List, Tuple


def compute_core_distances(dist_matrix: np.ndarray, min_pts: int) -> np.ndarray:
    """For each point, return the distance to its `min_pts`-th nearest neighbor.

    Note: rank 0 is the point itself (distance 0), so we take rank `min_pts`.
    """
    n = dist_matrix.shape[0]
    sorted_dists = np.sort(dist_matrix, axis=1)
    # Index `min_pts` because index 0 is the point itself
    return sorted_dists[:, min(min_pts, n - 1)]


def compute_reachability_matrix(dist_matrix: np.ndarray, core_dists: np.ndarray) -> np.ndarray:
    """rDist(p, q) = max(cDist(p), cDist(q), dist(p, q))."""
    n = len(core_dists)
    rdist = np.maximum(dist_matrix, core_dists[:, None])
    rdist = np.maximum(rdist, core_dists[None, :])
    return rdist


def ssdbscan_expand(
    rdist_matrix: np.ndarray,
    root_idx: int,
    label_map: Dict[int, int],
) -> Tuple[List[int], float]:
    """Expand from a single root point until a point with a different label is met.

    Returns the cluster member indices (after back-tracing) and the maximum
    edge length E_max along the back-traced segment.

    Args:
        rdist_matrix: precomputed pairwise reachability distances (n x n).
        root_idx: index of the root labeled point we are expanding from.
        label_map: dict mapping point_idx -> label (only contains labeled points).

    Algorithm:
        1. Maintain a min-heap keyed by current best `key` value (the smallest
           rDist needed to reach the point from the already-added set).
        2. Pop the closest point, add it to the expansion list.
        3. If the popped point has a label different from the root, stop and
           back-trace to find the longest edge along the path. All points
           strictly before that longest edge belong to the same cluster as the
           root.
        4. Otherwise, relax all unvisited neighbors.
    """
    n = rdist_matrix.shape[0]
    root_label = label_map[root_idx]

    INF = float("inf")
    key = np.full(n, INF)
    key[root_idx] = 0.0

    visited = np.zeros(n, dtype=bool)
    heap: List[Tuple[float, int, int]] = []  # (key, idx, parent_idx)
    heapq.heappush(heap, (0.0, root_idx, -1))

    # ordered list of (idx, edge_to_parent) for back-tracing
    expansion: List[Tuple[int, float]] = []

    terminator_label_seen = False
    while heap:
        edge_len, u, parent = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        expansion.append((u, edge_len))

        # Stop if we hit a point labeled differently from the root
        if u != root_idx and u in label_map and label_map[u] != root_label:
            terminator_label_seen = True
            break

        # Relax neighbors
        for v in range(n):
            if not visited[v]:
                d = rdist_matrix[u, v]
                if d < key[v]:
                    key[v] = d
                    heapq.heappush(heap, (d, v, u))

    if not terminator_label_seen:
        # We never hit another label; everything reachable is one cluster
        members = [idx for idx, _ in expansion]
        e_max = max((e for _, e in expansion if e > 0), default=0.0)
        return members, e_max

    # Back-trace: find the longest edge along the expansion path. Drop everything
    # from that edge onward (those points belong to a different cluster).
    edges = [(idx, edge) for idx, edge in expansion]
    longest_edge_pos = 0
    longest_edge_val = -1.0
    for i, (_, e) in enumerate(edges):
        if e > longest_edge_val:
            longest_edge_val = e
            longest_edge_pos = i

    # Members are everything strictly before the longest edge endpoint
    members = [idx for idx, _ in edges[:longest_edge_pos]]
    return members, longest_edge_val


def ssdbscan(
    X: np.ndarray,
    DN: Dict[int, int],
    distance_func,
    min_pts: int = 3,
) -> Tuple[Dict[int, int], np.ndarray, np.ndarray, np.ndarray]:
    """Run SSDBSCAN over all labeled normal points.

    Args:
        X: the data matrix (n, d).
        DN: dict mapping point_idx -> cluster_label (only labeled normal points).
        distance_func: an object with a `pairwise(X)` method (e.g. MahalanobisDistance).
        min_pts: SSDBSCAN's MinPts hyperparameter.

    Returns:
        cluster_assignments: dict point_idx -> cluster_label for the points
            successfully clustered. Points not in this dict are "unclustered" and
            will be classified later by the SSDBCODI step.
        dist_matrix: precomputed pairwise distance matrix (n, n).
        rdist_matrix: precomputed reachability matrix (n, n).
        core_dists: precomputed core distances (n,).
    """
    n = len(X)
    dist_matrix = distance_func.pairwise(X)
    core_dists = compute_core_distances(dist_matrix, min_pts)
    rdist_matrix = compute_reachability_matrix(dist_matrix, core_dists)

    cluster_assignments: Dict[int, int] = {}

    # Sort labeled points so the assignment order is deterministic
    for root_idx in sorted(DN.keys()):
        members, _ = ssdbscan_expand(rdist_matrix, root_idx, DN)
        root_label = DN[root_idx]
        for m in members:
            # Don't overwrite — first labeled root wins for contested points
            if m not in cluster_assignments:
                cluster_assignments[m] = root_label

    return cluster_assignments, dist_matrix, rdist_matrix, core_dists
