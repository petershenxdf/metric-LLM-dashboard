"""Enum of all intent types the chatbox can recognize."""
from enum import Enum


class IntentType(str, Enum):
    MUST_LINK = "must_link"
    CANNOT_LINK = "cannot_link"
    TRIPLET = "triplet"
    CLUSTER_COUNT = "cluster_count"
    OUTLIER_LABEL = "outlier_label"
    FEATURE_HINT = "feature_hint"
    CLUSTER_MERGE = "cluster_merge"
    REASSIGN = "reassign"
    OFF_TOPIC = "off_topic"
    VAGUE = "vague"
