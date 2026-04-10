"""Constraint completeness validators.

Each validator returns (is_valid, error_message). The chat service uses these
to decide whether to ask the user a follow-up question or to submit the
constraint.
"""
from typing import Tuple

from .schemas import (
    Constraint,
    MustLink,
    CannotLink,
    Triplet,
    ClusterCount,
    OutlierLabel,
    FeatureHint,
    ClusterMerge,
    Reassign,
)


def validate(constraint: Constraint, n_points: int) -> Tuple[bool, str]:
    """Dispatch to the right validator and return (ok, message)."""
    if isinstance(constraint, MustLink):
        return _validate_must_link(constraint, n_points)
    if isinstance(constraint, CannotLink):
        return _validate_cannot_link(constraint, n_points)
    if isinstance(constraint, Triplet):
        return _validate_triplet(constraint, n_points)
    if isinstance(constraint, ClusterCount):
        return _validate_cluster_count(constraint)
    if isinstance(constraint, OutlierLabel):
        return _validate_outlier_label(constraint, n_points)
    if isinstance(constraint, FeatureHint):
        return _validate_feature_hint(constraint)
    if isinstance(constraint, ClusterMerge):
        return _validate_cluster_merge(constraint)
    if isinstance(constraint, Reassign):
        return _validate_reassign(constraint, n_points)
    return False, "Unknown constraint type"


def _ids_in_range(ids, n_points):
    return all(0 <= i < n_points for i in ids)


def _validate_must_link(c: MustLink, n_points: int):
    if len(c.point_ids) < 2:
        return False, "Need at least 2 points for a must-link"
    if not _ids_in_range(c.point_ids, n_points):
        return False, "Point IDs out of range"
    return True, ""


def _validate_cannot_link(c: CannotLink, n_points: int):
    if not c.group_a or not c.group_b:
        return False, "Both groups must be non-empty"
    if not _ids_in_range(c.group_a + c.group_b, n_points):
        return False, "Point IDs out of range"
    if set(c.group_a) & set(c.group_b):
        return False, "Groups overlap"
    return True, ""


def _validate_triplet(c: Triplet, n_points: int):
    ids = [c.anchor, c.positive, c.negative]
    if any(i < 0 for i in ids):
        return False, "Triplet missing anchor / positive / negative"
    if not _ids_in_range(ids, n_points):
        return False, "Point IDs out of range"
    if len(set(ids)) != 3:
        return False, "Triplet members must be distinct"
    return True, ""


def _validate_cluster_count(c: ClusterCount):
    if c.target_k < 1:
        return False, "target_k must be >= 1"
    if c.scope not in ("all", "selected", "unselected"):
        return False, f"Invalid scope: {c.scope}"
    return True, ""


def _validate_outlier_label(c: OutlierLabel, n_points: int):
    if not c.point_ids:
        return False, "No points specified"
    if not _ids_in_range(c.point_ids, n_points):
        return False, "Point IDs out of range"
    return True, ""


def _validate_feature_hint(c: FeatureHint):
    if not c.feature_name:
        return False, "feature_name is required"
    if c.direction not in ("increase", "decrease", "ignore"):
        return False, f"Invalid direction: {c.direction}"
    if c.magnitude not in ("slight", "moderate", "strong"):
        return False, f"Invalid magnitude: {c.magnitude}"
    return True, ""


def _validate_cluster_merge(c: ClusterMerge):
    if len(c.cluster_ids) < 2:
        return False, "Need at least 2 cluster ids to merge"
    return True, ""


def _validate_reassign(c: Reassign, n_points: int):
    if not c.point_ids:
        return False, "No points specified"
    if not _ids_in_range(c.point_ids, n_points):
        return False, "Point IDs out of range"
    if c.target_cluster_id < 0:
        return False, "target_cluster_id must be >= 0"
    return True, ""
