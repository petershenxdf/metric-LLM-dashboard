"""Tests for the rule-based intent classifier."""
from app.domain.intent.rule_classifier import RuleClassifier
from app.domain.intent.intent_types import IntentType


def test_must_link_phrases():
    rc = RuleClassifier()
    phrases = [
        "these points are the same cluster",
        "they belong together",
        "group these together",
    ]
    for phrase in phrases:
        intent, _ = rc.classify(phrase)
        assert intent == IntentType.MUST_LINK, f"Failed on: {phrase}"


def test_outlier_phrases():
    rc = RuleClassifier()
    phrases = [
        "these are outliers",
        "that is an anomaly",
        "this looks abnormal",
        "just noise",
    ]
    for phrase in phrases:
        intent, _ = rc.classify(phrase)
        assert intent == IntentType.OUTLIER_LABEL, f"Failed on: {phrase}"


def test_cluster_count_with_number():
    rc = RuleClassifier()
    intent, slots = rc.classify("split into 4 clusters")
    assert intent == IntentType.CLUSTER_COUNT
    assert slots["target_k"] == 4

    intent, slots = rc.classify("should be 3 groups")
    assert intent == IntentType.CLUSTER_COUNT
    assert slots["target_k"] == 3


def test_cluster_merge():
    rc = RuleClassifier()
    intent, slots = rc.classify("merge cluster 2 and 5")
    assert intent == IntentType.CLUSTER_MERGE
    assert slots["cluster_ids"] == [2, 5]


def test_feature_hint():
    rc = RuleClassifier()
    intent, _ = rc.classify("the color feature is not important")
    assert intent == IntentType.FEATURE_HINT


def test_unrecognized_returns_none():
    rc = RuleClassifier()
    intent, _ = rc.classify("what is the weather today")
    assert intent is None

    intent, _ = rc.classify("hello there")
    assert intent is None


def test_empty_input():
    rc = RuleClassifier()
    intent, _ = rc.classify("")
    assert intent is None
    intent, _ = rc.classify("   ")
    assert intent is None
