"""Constraint schemas — typed dataclasses for the 8 constraint types.

These mirror the JSON shapes the LLM is asked to produce. Each class has
to_dict() / from_dict() so we can round-trip through HTTP and storage.

The base Constraint class holds common metadata (timestamp, source, confidence).
"""
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class Constraint:
    """Base class — never instantiate directly."""
    type: str = ""
    confidence: str = "explicit"  # explicit | suggested
    source: str = "user"          # user | rule | llm
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MustLink(Constraint):
    point_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.type = "must_link"


@dataclass
class CannotLink(Constraint):
    group_a: List[int] = field(default_factory=list)
    group_b: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.type = "cannot_link"


@dataclass
class Triplet(Constraint):
    anchor: int = -1
    positive: int = -1
    negative: int = -1

    def __post_init__(self):
        self.type = "triplet"


@dataclass
class ClusterCount(Constraint):
    scope: str = "all"          # all | selected | unselected
    target_k: int = 2

    def __post_init__(self):
        self.type = "cluster_count"


@dataclass
class OutlierLabel(Constraint):
    point_ids: List[int] = field(default_factory=list)
    is_outlier: bool = True

    def __post_init__(self):
        self.type = "outlier_label"


@dataclass
class FeatureHint(Constraint):
    feature_name: str = ""
    direction: str = "decrease"  # increase | decrease | ignore
    magnitude: str = "moderate"  # slight | moderate | strong

    def __post_init__(self):
        self.type = "feature_hint"


@dataclass
class ClusterMerge(Constraint):
    cluster_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.type = "cluster_merge"


@dataclass
class Reassign(Constraint):
    point_ids: List[int] = field(default_factory=list)
    target_cluster_id: int = 0

    def __post_init__(self):
        self.type = "reassign"


# Mapping for the from_dict factory
_TYPE_MAP = {
    "must_link": MustLink,
    "cannot_link": CannotLink,
    "triplet": Triplet,
    "cluster_count": ClusterCount,
    "outlier_label": OutlierLabel,
    "feature_hint": FeatureHint,
    "cluster_merge": ClusterMerge,
    "reassign": Reassign,
}


def constraint_from_dict(d: Dict[str, Any]) -> Constraint:
    """Build the right Constraint subclass from a dict (e.g. JSON from LLM)."""
    ctype = d.get("type")
    if ctype not in _TYPE_MAP:
        raise ValueError(f"Unknown constraint type: {ctype}")

    cls = _TYPE_MAP[ctype]
    # Strip the 'type' key — __post_init__ sets it
    payload = {k: v for k, v in d.items() if k != "type"}

    # Filter to only fields the dataclass actually has
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in payload.items() if k in valid_fields}

    return cls(**filtered)
