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
    ) -> Dict[str, Any]:
        state = self.session_service.get(session_id)
        if state is None:
            return {"error": f"Session {session_id} not found"}

        # 1. Record the user message
        state.chat_history.append({"role": "user", "content": user_text})

        # 2. Try the rule-based classifier first
        rule_intent, rule_slots = self.rule_classifier.classify(user_text)
        if rule_intent is not None:
            response = self._handle_rule_result(
                state, user_text, selected_ids, rule_intent, rule_slots
            )
        else:
            # 3. Fall back to the LLM
            response = self._handle_llm_result(state, user_text, selected_ids)

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
        self, state, user_text, selected_ids, intent, slots
    ) -> Dict[str, Any]:
        """The rule classifier recognized an intent. Build the constraint.

        For rule-matched intents that need point IDs (must_link, outlier_label,
        etc.), we pull the IDs from `selected_ids`. If the user selected nothing
        and the intent needs points, ask a follow-up instead.
        """
        constraint_dict = self._build_constraint_from_rule(
            intent, slots, selected_ids
        )

        if constraint_dict is None:
            return {
                "intent": intent.value,
                "complete": False,
                "constraint": None,
                "followup_question": self._rule_followup_for(intent),
                "assistant_message": self._rule_followup_for(intent),
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
        self, intent: IntentType, slots: Dict[str, Any], selected_ids: List[int]
    ) -> Dict[str, Any]:
        """Combine rule-extracted slots with the user's selection state."""
        if intent == IntentType.MUST_LINK:
            if len(selected_ids) < 2:
                return None
            return {
                "type": "must_link",
                "point_ids": list(selected_ids),
                "confidence": "explicit",
                "source": "rule",
            }
        if intent == IntentType.OUTLIER_LABEL:
            if not selected_ids:
                return None
            return {
                "type": "outlier_label",
                "point_ids": list(selected_ids),
                "is_outlier": slots.get("is_outlier", True),
                "confidence": "explicit",
                "source": "rule",
            }
        if intent == IntentType.CLUSTER_COUNT:
            if "target_k" not in slots:
                return None
            return {
                "type": "cluster_count",
                "scope": "unselected" if selected_ids else "all",
                "target_k": slots["target_k"],
                "confidence": "explicit",
                "source": "rule",
            }
        if intent == IntentType.CLUSTER_MERGE:
            if "cluster_ids" not in slots:
                return None
            return {
                "type": "cluster_merge",
                "cluster_ids": slots["cluster_ids"],
                "confidence": "explicit",
                "source": "rule",
            }
        # For cannot_link / triplet / feature_hint the rule path alone isn't
        # enough -- we need extra info from the LLM. Return None to fall
        # through to a follow-up question.
        return None

    def _rule_followup_for(self, intent: IntentType) -> str:
        if intent == IntentType.MUST_LINK:
            return "Please select at least 2 points on the scatterplot first."
        if intent == IntentType.OUTLIER_LABEL:
            return "Please select the points you want to mark as outliers."
        if intent == IntentType.CANNOT_LINK:
            return (
                "Which points should NOT be together? Select one group, "
                "then tell me which cluster they should be separated from."
            )
        if intent == IntentType.TRIPLET:
            return (
                "Which three points? I need an anchor, a point it should be "
                "close to, and a point it should be far from."
            )
        if intent == IntentType.FEATURE_HINT:
            return "Which feature, and should I reduce its importance or ignore it entirely?"
        return "Could you give me a bit more detail?"

    def _rule_confirmation_for(self, intent: IntentType, constraint: Dict[str, Any]) -> str:
        if intent == IntentType.MUST_LINK:
            n = len(constraint.get("point_ids", []))
            return f"Got it -- marking these {n} points as the same cluster."
        if intent == IntentType.OUTLIER_LABEL:
            n = len(constraint.get("point_ids", []))
            is_out = constraint.get("is_outlier", True)
            verb = "marking" if is_out else "unmarking"
            return f"Got it -- {verb} {n} points as outliers."
        if intent == IntentType.CLUSTER_COUNT:
            k = constraint.get("target_k")
            scope = constraint.get("scope", "all")
            return f"Got it -- aiming for {k} clusters on the {scope} points."
        if intent == IntentType.CLUSTER_MERGE:
            ids = constraint.get("cluster_ids", [])
            return f"Got it -- merging clusters {ids} into one."
        return "Got it."

    # ------------------------------------------------------------------
    # LLM-path handler
    # ------------------------------------------------------------------

    def _handle_llm_result(self, state, user_text, selected_ids) -> Dict[str, Any]:
        cluster_summary = self._build_cluster_summary(state)
        result = self.llm_classifier.classify(
            user_text=user_text,
            selected_point_ids=selected_ids,
            cluster_summary=cluster_summary,
            n_points=state.n_points(),
        )

        intent = result.get("intent", "vague")
        complete = result.get("complete", False)
        constraint_dict = result.get("constraint")
        followup = result.get("followup_question")
        confirmation = result.get("confirmation_message")

        if intent in ("off_topic", "vague") or not complete or constraint_dict is None:
            return {
                "intent": intent,
                "complete": False,
                "constraint": None,
                "followup_question": followup,
                "assistant_message": followup or "Could you clarify what you'd like to do?",
                "source": "llm",
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
