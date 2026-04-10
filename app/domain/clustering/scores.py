"""The three outlier-ness scores from the SSDBCODI paper.

These are computed once per pipeline run after SSDBSCAN has produced rDist and
the labeled sets DN, DO are known. All score values are normalized to [0, 1]
via exponential transforms so they can be combined into the total tScore.
"""
import numpy as np
from typing import Dict, Set


def compute_reachability_score(
    rdist_matrix: np.ndarray,
    DN: Dict[int, int],
) -> np.ndarray:
    """rScore(q) = exp(-min over labeled-normals p of E_max(q, p)).

    We approximate E_max(q, p) by the rDist between q and the nearest labeled
    normal point — this is the simplest practical surrogate, and it agrees with
    the spirit of the paper (a point that is hard to reach from any normal
    cluster is likely an outlier).

    A high rScore means the point is easy to reach from a labeled cluster
    (looks normal). A low rScore means it is far from any normal cluster
    (looks like an outlier).
    """
    n = rdist_matrix.shape[0]
    if not DN:
        return np.ones(n)

    dn_indices = np.array(sorted(DN.keys()))
    # rDist from every point to every labeled normal
    sub = rdist_matrix[:, dn_indices]
    nearest_normal_rdist = sub.min(axis=1)
    return np.exp(-nearest_normal_rdist)


def compute_local_density_score(
    rdist_matrix: np.ndarray,
    min_pts: int,
) -> np.ndarray:
    """lScore(q) = exp(-LD(q)), where LD(q) is the average rDist to its
    `min_pts` nearest neighbors.

    A high lScore means the point sits in a dense neighborhood (normal). A low
    lScore means a sparse neighborhood (likely outlier).
    """
    n = rdist_matrix.shape[0]
    if n <= 1:
        return np.ones(n)

    # Sort each row, take the smallest `min_pts` non-self distances
    sorted_rdist = np.sort(rdist_matrix, axis=1)
    # Skip column 0 (self), take next min_pts columns
    k = min(min_pts, n - 1)
    avg_rdist = sorted_rdist[:, 1:k + 1].mean(axis=1)
    return np.exp(-avg_rdist)


def compute_similarity_score(
    dist_matrix: np.ndarray,
    DO: Set[int],
) -> np.ndarray:
    """simScore(q) = exp(-min over labeled-outliers o of dist(q, o)).

    A high simScore means the point is close to a known outlier (looks
    outlier-like). A low simScore means it is far from any known outlier.

    If there are no labeled outliers, every point gets simScore = 0.
    """
    n = dist_matrix.shape[0]
    if not DO:
        return np.zeros(n)

    do_indices = np.array(sorted(DO))
    sub = dist_matrix[:, do_indices]
    nearest_outlier_dist = sub.min(axis=1)
    return np.exp(-nearest_outlier_dist)


def compute_total_score(
    rscore: np.ndarray,
    lscore: np.ndarray,
    simscore: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """tScore from Equation 7 of the paper.

    tScore(q) = alpha * (1 - rScore) + beta * (1 - lScore) + (1 - alpha - beta) * simScore

    Higher tScore means the point looks more like an outlier.
    Constraints: alpha, beta in [0, 1] and alpha + beta <= 1.
    """
    if alpha < 0 or beta < 0 or alpha + beta > 1:
        raise ValueError(f"Invalid alpha={alpha}, beta={beta}; need both >= 0 and sum <= 1")
    gamma = 1.0 - alpha - beta
    return alpha * (1.0 - rscore) + beta * (1.0 - lscore) + gamma * simscore
