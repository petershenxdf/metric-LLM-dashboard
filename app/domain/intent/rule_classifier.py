"""Rule-based fast-path classifier.

Tries to recognize the user's intent from simple keyword patterns. If it can,
the system skips the LLM call entirely -- saving latency and tokens.

The classifier returns either:
- (IntentType, slots_dict) when it is confident, or
- (None, None) when the LLM should take over.

slots_dict contains whatever fields the rule could extract; the chat service
fills in selected_point_ids etc. before building the final Constraint.
"""
import re
from typing import Optional, Tuple, Dict, Any

from .intent_types import IntentType


# Each pattern is (regex, intent, slot_extractor). The slot_extractor is a
# callable receiving the regex match and returning a dict of partial slots.
# Order matters: more specific patterns come first.
_PATTERNS = [
    # cluster_count -- match first because it has the most specific pattern
    (
        re.compile(
            r"\b(?:into|split\s+into|should\s*be|need|want)\s+(\d+)\s*(?:class|cluster|group)s?\b",
            re.IGNORECASE,
        ),
        IntentType.CLUSTER_COUNT,
        lambda m: {"target_k": int(m.group(1))},
    ),
    # cluster_merge -- explicit "merge cluster X and Y"
    (
        re.compile(
            r"\bmerge\s+(?:clusters?|groups?)\s+(\d+)\s+and\s+(\d+)\b",
            re.IGNORECASE,
        ),
        IntentType.CLUSTER_MERGE,
        lambda m: {"cluster_ids": [int(m.group(1)), int(m.group(2))]},
    ),
    # must_link
    (
        re.compile(
            r"\b(same\s*(?:class|cluster|group)|one\s*(?:class|cluster|group)|"
            r"belong\s*together|group\s*these\s*together)\b",
            re.IGNORECASE,
        ),
        IntentType.MUST_LINK,
        lambda m: {},
    ),
    # cannot_link
    (
        re.compile(
            r"\b(should\s*not\s*be\s*together|separate|different\s*(?:class|cluster|group)|"
            r"not\s*the\s*same\s*(?:class|cluster|group))\b",
            re.IGNORECASE,
        ),
        IntentType.CANNOT_LINK,
        lambda m: {},
    ),
    # triplet
    (
        re.compile(
            r"\b(more\s*similar\s*to|closer\s*to|more\s*like)\b",
            re.IGNORECASE,
        ),
        IntentType.TRIPLET,
        lambda m: {},
    ),
    # outlier_label
    (
        re.compile(
            r"\b(outliers?|anomal(?:y|ies|ous)|noise|abnormal)\b",
            re.IGNORECASE,
        ),
        IntentType.OUTLIER_LABEL,
        lambda m: {"is_outlier": True},
    ),
    # feature_hint
    (
        re.compile(
            r"\b(?:feature|dimension|attribute|column)\b.*\b"
            r"(?:not\s*important|ignore|unimportant|less\s*important|irrelevant)\b",
            re.IGNORECASE,
        ),
        IntentType.FEATURE_HINT,
        lambda m: {"direction": "decrease"},
    ),
]


class RuleClassifier:
    def classify(self, text: str) -> Tuple[Optional[IntentType], Dict[str, Any]]:
        """Try to classify with regex rules. Return (None, {}) if no match."""
        if not text or not text.strip():
            return None, {}

        for pattern, intent, extractor in _PATTERNS:
            m = pattern.search(text)
            if m:
                slots = extractor(m)
                return intent, slots

        return None, {}
