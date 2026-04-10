"""Tests for constraint schemas, router, and validators."""
import pytest

from app.domain.constraints.schemas import (
    MustLink,
    CannotLink,
    Triplet,
    ClusterCount,
    OutlierLabel,
    FeatureHint,
    ClusterMerge,
    Reassign,
    constraint_from_dict,
)
from app.domain.constraints.router import route_constraint, ChannelType
from app.domain.constraints.validators import validate


def test_constraint_from_dict_roundtrip():
    d = {"type": "must_link", "point_ids": [1, 2, 3], "confidence": "explicit"}
    c = constraint_from_dict(d)
    assert isinstance(c, MustLink)
    assert c.point_ids == [1, 2, 3]
    assert c.type == "must_link"


def test_all_types_parseable():
    """Every constraint type should roundtrip through from_dict / to_dict."""
    cases = [
        {"type": "must_link", "point_ids": [1, 2]},
        {"type": "cannot_link", "group_a": [1], "group_b": [2]},
        {"type": "triplet", "anchor": 0, "positive": 1, "negative": 2},
        {"type": "cluster_count", "scope": "all", "target_k": 3},
        {"type": "outlier_label", "point_ids": [5], "is_outlier": True},
        {"type": "feature_hint", "feature_name": "x1", "direction": "decrease", "magnitude": "moderate"},
        {"type": "cluster_merge", "cluster_ids": [0, 1]},
        {"type": "reassign", "point_ids": [10], "target_cluster_id": 2},
    ]
    for d in cases:
        c = constraint_from_dict(d)
        assert c.type == d["type"]
        # Round-trip back to dict
        out = c.to_dict()
        assert out["type"] == d["type"]


def test_router_channels():
    assert route_constraint(MustLink(point_ids=[1, 2])) == ChannelType.BOTH
    assert route_constraint(CannotLink(group_a=[1], group_b=[2])) == ChannelType.METRIC
    assert route_constraint(Triplet(anchor=0, positive=1, negative=2)) == ChannelType.METRIC
    assert route_constraint(FeatureHint(feature_name="f0")) == ChannelType.METRIC
    assert route_constraint(ClusterCount(target_k=3)) == ChannelType.LABEL
    assert route_constraint(OutlierLabel(point_ids=[1])) == ChannelType.LABEL
    assert route_constraint(ClusterMerge(cluster_ids=[0, 1])) == ChannelType.LABEL
    assert route_constraint(Reassign(point_ids=[1], target_cluster_id=2)) == ChannelType.LABEL


def test_validator_must_link():
    ok, _ = validate(MustLink(point_ids=[1, 2, 3]), n_points=10)
    assert ok

    ok, msg = validate(MustLink(point_ids=[1]), n_points=10)
    assert not ok

    ok, msg = validate(MustLink(point_ids=[1, 99]), n_points=10)
    assert not ok


def test_validator_triplet():
    ok, _ = validate(Triplet(anchor=0, positive=1, negative=2), n_points=10)
    assert ok

    # Duplicate members
    ok, _ = validate(Triplet(anchor=0, positive=0, negative=1), n_points=10)
    assert not ok


def test_validator_feature_hint():
    ok, _ = validate(
        FeatureHint(feature_name="x1", direction="decrease", magnitude="moderate"),
        n_points=10,
    )
    assert ok

    ok, _ = validate(
        FeatureHint(feature_name="x1", direction="invalid", magnitude="moderate"),
        n_points=10,
    )
    assert not ok
