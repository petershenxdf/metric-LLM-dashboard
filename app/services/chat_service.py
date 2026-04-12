"""Chat service -- processes one user message end to end.

Flow:
    1. Append the user message to the session's chat history.
    2. Try the rule-based fast path. If it returns a recognized intent, build
       a constraint directly.
    3. Otherwise, call the LLM-based classifier.
    4. If the classifier returns an incomplete result (need follow-up), append
       the follow-up question to the chat history and return it.
    5. If the classifier returns a complete constraint, validate it. If valid,
       return it to the frontend as a "ready to submit" suggestion with a
       confirmation message. The frontend can then auto-submit or ask the
       user to confirm.
"""
from typing import Dict, Any, List

from config.config import Config
from app.services.session_service import SessionService
from app.domain.intent.rule_classifier import RuleClassifier
from app.domain.intent.llm_classifier import LLMIntentClassifier
from app.domain.intent.intent_types import IntentType
from app.domain.constraints.schemas import (
    constraint_from_dict,
    MustLink,
    OutlierLabel,
)
from app.domain.constraints.validators import validate
from app.infrastructure.llm.base import LLMClient
from app.infrastructure.debug.logger import get_logger


logger = get_logger("chat")


class ChatService:
    def __init__(
        self,
        session_service: SessionService,
        llm_client: LLMClient,
        config: Config,
    ):
        self.session_service = session_service
        self.config = config
        self.rule_classifier = RuleClassifier()
        self.llm_classifier = LLMIntentClassifier(
            llm_client=llm_client,
            system_prompt_path=config.prompts_folder / "system_prompt.txt",
            few_shot_path=config.prompts_folder / "few_shot_examples.json",
        )

    def process_message(
        self,
        session_id: str,
        user_text: str,
        selected_ids: List[int],
        selection_groups: List[List[int]] = None,
    ) -> Dict[str, Any]:
        state = self.session_service.get(session_id)
        if state is None:
            return {"error": f"Session {session_id} not found"}

        selection_groups = selection_groups or []

        # 1. Record the user message
        state.chat_history.append({"role": "user", "content": user_text})

        # 2. Try the rule-based classifier first
        rule_intent, rule_slots = self.rule_classifier.classify(user_text)
        if rule_intent is not None:
            response = self._handle_rule_result(
                state, user_text, selected_ids, rule_intent, rule_slots,
                selection_groups,
            )
        else:
            # 3. Fall back to the LLM
            response = self._handle_llm_result(
                state, user_text, selected_ids, selection_groups
            )

        # 4. Record the assistant message in history
        assistant_text = response.get("assistant_message", "")
        if assistant_text:
            state.chat_history.append({"role": "assistant", "content": assistant_text})

        self.session_service.save(state)
        return response

    # ------------------------------------------------------------------
    # Rule-path handler
    # ------------------------------------------------------------------

    def _handle_rule_result(
        self, state, user_text, selected_ids, intent, slots, selection_groups
    ) -> Dict[str, Any]:
        """The rule classifier recognized an intent. Build the constraint.

        For rule-matched intents that need point IDs (must_link, outlier_label,
        etc.), we pull the IDs from `selected_ids`. If the user selected nothing
        and the intent needs points, ask a follow-up instead.
        """
        constraint_dict, build_error, build_note = self._build_constraint_from_rule(
            intent, slots, selected_ids, selection_groups
        )

        if constraint_dict is None:
            msg = build_error or self._rule_followup_for(intent)
            return {
                "intent": intent.value,
                "complete": False,
                "constraint": None,
                "followup_question": msg,
                "assistant_message": msg,
                "source": "rule",
            }

        # Try to build and validate the constraint
        try:
            constraint = constraint_from_dict(constraint_dict)
        except Exception as e:
            return {
                "intent": intent.value,
                "complete": False,
                "constraint": None,
                "followup_question": f"I could not build a valid constraint: {e}",
                "assistant_message": f"I could not build a valid constraint: {e}",
                "source": "rule",
            }

        ok, msg = validate(constraint, state.n_points())
        if not ok:
            return {
                "intent": intent.value,
                "complete": False,
                "constraint": None,
                "followup_question": msg,
                "assistant_message": msg,
                "source": "rule",
            }

        confirmation = self._rule_confirmation_for(intent, constraint_dict)
        if build_note:
            confirmation = f"{confirmation} ({build_note})"
        return {
            "intent": intent.value,
            "complete": True,
            "constraint": constraint_dict,
            "followup_question": None,
            "confirmation_message": confirmation,
            "assistant_message": confirmation,
            "source": "rule",
        }

    def _build_constraint_from_rule(
        self,
        intent: IntentType,
        slots: Dict[str, Any],
        selected_ids: List[int],
        selection_groups: List[List[int]],
    ):
        """Combine rule-extracted slots with the user's selection state.

        Returns a 3-tuple ``(constraint_dict, error, note)``:
        - ``constraint_dict``: the built constraint, or None on failure.
        - ``error``: a human-readable reason when the build failed (so the
          chat can surface it instead of a generic follow-up).
        - ``note``: a non-fatal warning that should be appended to the
          confirmation (e.g. "removed 3 overlapping points").

        ``selection_groups`` is a list of point-id lists staged by the user
        via the chatbox "Add as group" button. It unlocks cannot-link (needs
        2 groups) and triplet (needs 3 groups, each with a single point).
        """
        if intent == IntentType.MUST_LINK:
            ids = self._flatten_groups(selection_groups) or list(selected_ids)
            if len(ids) < 2:
                return None, (
                    "Must-link needs at least 2 points. Select some points "
                    "first, or add multiple groups."
                ), None
            return {
                "type": "must_link",
                "point_ids": ids,
                "confidence": "explicit",
                "source": "rule",
            }, None, None

        if intent == IntentType.CANNOT_LINK:
            if len(selection_groups) < 2:
                return None, (
                    "Cannot-link needs two groups. Select the first set and "
                    "click 'Add as group', then select the second set and "
                    "click 'Add as group' again."
                ), None
            group_a = [int(x) for x in selection_groups[0]]
            group_b = [int(x) for x in selection_groups[1]]
            if not group_a or not group_b:
                return None, "Both groups must contain at least one point.", None

            # Auto-resolve overlap: points in both groups are ambiguous, so
            # strip them from group_b and warn. If that leaves group_b empty,
            # the user really did stage the same set twice -- report that.
            overlap = set(group_a) & set(group_b)
            note = None
            if overlap:
                group_b = [pid for pid in group_b if pid not in overlap]
                if not group_b:
                    return None, (
                        f"Group A and Group B overlap completely "
                        f"({len(overlap)} shared points). Select a different "
                        f"second group before asking for cannot-link."
                    ), None
                note = (
                    f"removed {len(overlap)} overlapping point"
                    f"{'s' if len(overlap) != 1 else ''} from Group B"
                )
            return {
                "type": "cannot_link",
                "group_a": group_a,
                "group_b": group_b,
                "confidence": "explicit",
                "source": "rule",
            }, None, note

        if intent == IntentType.TRIPLET:
            if len(selection_groups) < 3:
                return None, (
                    "Triplet needs three groups (anchor, positive, negative). "
                    "Add each as a separate group."
                ), None
            if not all(selection_groups[i] for i in range(3)):
                return None, "Each triplet group must contain at least one point.", None
            anchor = int(selection_groups[0][0])
            positive = int(selection_groups[1][0])
            negative = int(selection_groups[2][0])
            if len({anchor, positive, negative}) < 3:
                return None, (
                    "Triplet points must be distinct (anchor, positive, "
                    "negative cannot repeat)."
                ), None
            return {
                "type": "triplet",
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "confidence": "explicit",
                "source": "rule",
            }, None, None

        if intent == IntentType.OUTLIER_LABEL:
            ids = self._flatten_groups(selection_groups) or list(selected_ids)
            if not ids:
                return None, (
                    "Please select the points you want to mark as outliers."
                ), None
            return {
                "type": "outlier_label",
                "point_ids": ids,
                "is_outlier": slots.get("is_outlier", True),
                "confidence": "explicit",
                "source": "rule",
            }, None, None

        if intent == IntentType.CLUSTER_COUNT:
            if "target_k" not in slots:
                return None, "How many clusters should I target?", None
            return {
                "type": "cluster_count",
                "scope": "unselected" if selected_ids else "all",
                "target_k": slots["target_k"],
                "confidence": "explicit",
                "source": "rule",
            }, None, None

        if intent == IntentType.CLUSTER_MERGE:
            if "cluster_ids" not in slots:
                return None, "Which clusters should I merge?", None
            return {
                "type": "cluster_merge",
                "cluster_ids": slots["cluster_ids"],
                "confidence": "explicit",
                "source": "rule",
            }, None, None

        # feature_hint still needs the LLM because the rule only detects
        # intent -- it can't extract the feature name reliably.
        return None, None, None

    @staticmethod
    def _flatten_groups(selection_groups: List[List[int]]) -> List[int]:
        seen = set()
        out = []
        for g in selection_groups:
            for pid in g:
                if pid not in seen:
                    seen.add(pid)
                    out.append(int(pid))
        return out

    def _rule_followup_for(self, intent: IntentType) -> str:
        if intent == IntentType.MUST_LINK:
            return "Please select at least 2 points on the scatterplot first."
        if intent == IntentType.OUTLIER_LABEL:
            return "Please select the points you want to mark as outliers."
        if intent == IntentType.CANNOT_LINK:
            return (
                "Cannot-link needs two groups. Select the first set and click "
                "'Add as group', then select the second set and click 'Add as "
                "group' again. Then say 'these should not be together'."
            )
        if intent == IntentType.TRIPLET:
            return (
                "Triplet needs three single-point groups: anchor, positive, "
                "and negative. Select one point and add it as a group, then "
                "repeat twice more (order matters)."
            )
        if intent == IntentType.FEATURE_HINT:
            return "Which feature, and should I reduce its importance or ignore it entirely?"
        return "Could you give me a bit more detail?"

    def _rule_confirmation_for(self, intent: IntentType, constraint: Dict[str, Any]) -> str:
        if intent == IntentType.MUST_LINK:
            n = len(constraint.get("point_ids", []))
            return (
                f"Queued must-link on {n} points. Click 'Run clustering' "
                f"to apply."
            )
        if intent == IntentType.CANNOT_LINK:
            na = len(constraint.get("group_a", []))
            nb = len(constraint.get("group_b", []))
            return (
                f"Queued cannot-link ({na} vs {nb} points). Click 'Run "
                f"clustering' to apply."
            )
        if intent == IntentType.TRIPLET:
            return (
                "Queued triplet constraint. Click 'Run clustering' to apply."
            )
        if intent == IntentType.OUTLIER_LABEL:
            n = len(constraint.get("point_ids", []))
            is_out = constraint.get("is_outlier", True)
            verb = "mark" if is_out else "unmark"
            return (
                f"Queued: {verb} {n} points as outliers. Click 'Run "
                f"clustering' to apply."
            )
        if intent == IntentType.CLUSTER_COUNT:
            k = constraint.get("target_k")
            scope = constraint.get("scope", "all")
            return f"Queued cluster_count={k} on {scope}. Click 'Run clustering' to apply."
        if intent == IntentType.CLUSTER_MERGE:
            ids = constraint.get("cluster_ids", [])
            return f"Queued merge of clusters {ids}. Click 'Run clustering' to apply."
        return "Queued. Click 'Run clustering' to apply."

    # ------------------------------------------------------------------
    # LLM-path handler
    # ------------------------------------------------------------------

    def _handle_llm_result(
        self, state, user_text, selected_ids, selection_groups
    ) -> Dict[str, Any]:
        cluster_summary = self._build_cluster_summary(state)
        # Pass the flat union of staged groups as selected points so the LLM
        # has access to the same info that powers the rule path. If no groups
        # were staged, fall back to the raw selection.
        effective_ids = self._flatten_groups(selection_groups) or list(selected_ids)
        result = self.llm_classifier.classify(
            user_text=user_text,
            selected_point_ids=effective_ids,
            cluster_summary=cluster_summary,
            n_points=state.n_points(),
        )

        intent = result.get("intent", "vague")
        complete = result.get("complete", False)
        constraint_dict = result.get("constraint")
        followup = result.get("followup_question")
        confirmation = result.get("confirmation_message")
        llm_error = result.get("_error")

        if llm_error:
            logger.warning("LLM classifier error: %s", llm_error)

        if intent in ("off_topic", "vague") or not complete or constraint_dict is None:
            # If the LLM call itself failed, surface the real error to the user
            # instead of the generic "I had trouble understanding" message.
            if llm_error:
                assistant = f"LLM error: {llm_error}"
            else:
                assistant = followup or "Could you clarify what you'd like to do?"
            return {
                "intent": intent,
                "complete": False,
                "constraint": None,
                "followup_question": followup,
                "assistant_message": assistant,
                "source": "llm",
                "error_detail": llm_error,
            }

        # Validate the constraint before returning
        try:
            constraint_obj = constraint_from_dict(constraint_dict)
        except Exception as e:
            return {
                "intent": intent,
                "complete": False,
                "constraint": None,
                "followup_question": f"I could not parse that as a constraint: {e}",
                "assistant_message": f"I could not parse that as a constraint: {e}",
                "source": "llm",
            }

        ok, msg = validate(constraint_obj, state.n_points())
        if not ok:
            return {
                "intent": intent,
                "complete": False,
                "constraint": None,
                "followup_question": msg,
                "assistant_message": msg,
                "source": "llm",
            }

        return {
            "intent": intent,
            "complete": True,
            "constraint": constraint_dict,
            "followup_question": None,
            "confirmation_message": confirmation,
            "assistant_message": confirmation or "Got it.",
            "source": "llm",
        }

    def _build_cluster_summary(self, state) -> Dict[str, Any]:
        if state.current_clusters is None:
            return {"status": "not_clustered"}
        cluster_sizes = {}
        for c in state.current_clusters:
            ci = int(c)
            cluster_sizes[ci] = cluster_sizes.get(ci, 0) + 1
        return {
            "n_clusters": len([k for k in cluster_sizes.keys() if k >= 0]),
            "cluster_sizes": cluster_sizes,
            "n_outliers": int(sum(state.current_outliers)),
        }
