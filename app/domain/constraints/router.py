"""Decide which channel(s) a constraint should be applied to.

Two channels:
- LABEL channel: directly modifies DN/DO (the labeled normal / outlier sets
  that SSDBCODI consumes).
- METRIC channel: feeds the constraint into the metric learner so M is updated
  before SSDBCODI re-runs.

Some constraints affect both channels.
"""
from enum import Enum

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


class ChannelType(str, Enum):
    LABEL = "label"
    METRIC = "metric"
    BOTH = "both"
    NONE = "none"


def route_constraint(constraint: Constraint) -> ChannelType:
    """Return which channel a constraint should be applied to.

    The mapping (also documented in the design doc):

        must_link       -> METRIC (and BOTH if user is also assigning labels)
        cannot_link     -> METRIC
        triplet         -> METRIC
        cluster_count   -> LABEL  (passed to SSDBCODI as a hint)
        outlier_label   -> LABEL  (modifies DO)
        feature_hint    -> METRIC (modifies M directly)
        cluster_merge   -> LABEL  (modifies DN cluster ids)
        reassign        -> LABEL  (modifies DN)
    """
    if isinstance(constraint, MustLink):
        # must_link informs the metric AND can populate DN if a target cluster
        # is implied. For now, treat as BOTH so the pipeline service can
        # decide whether label assignment makes sense.
        return ChannelType.BOTH
    if isinstance(constraint, CannotLink):
        return ChannelType.METRIC
    if isinstance(constraint, Triplet):
        return ChannelType.METRIC
    if isinstance(constraint, ClusterCount):
        return ChannelType.LABEL
    if isinstance(constraint, OutlierLabel):
        return ChannelType.LABEL
    if isinstance(constraint, FeatureHint):
        return ChannelType.METRIC
    if isinstance(constraint, ClusterMerge):
        return ChannelType.LABEL
    if isinstance(constraint, Reassign):
        return ChannelType.LABEL
    return ChannelType.NONE
