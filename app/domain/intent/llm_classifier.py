"""LLM-based intent classifier.

When the rule-based fast path fails, we fall back to an LLM. This class
constructs the prompt, calls the injected LLMClient, and parses the JSON
response into a structured result the chat service can act on.

The LLM is asked to return JSON only. We do best-effort cleanup (strip
markdown fences, find the first { ... } block) before parsing.
"""
import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.infrastructure.llm.base import LLMClient


class LLMIntentClassifier:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt_path: Path,
        few_shot_path: Optional[Path] = None,
    ):
        self.llm = llm_client
        self.system_prompt_template = self._load_text(system_prompt_path)
        self.few_shot_examples = self._load_json(few_shot_path) if few_shot_path else []

    @staticmethod
    def _load_text(path: Path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _load_json(path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def classify(
        self,
        user_text: str,
        selected_point_ids: List[int],
        cluster_summary: Dict[str, Any],
        n_points: int,
    ) -> Dict[str, Any]:
        """Call the LLM and parse the JSON response.

        Returns a dict with keys: intent, complete, constraint, followup_question,
        confirmation_message. On parse failure, returns a 'vague' fallback.
        """
        system_prompt = self.system_prompt_template.format(
            selected_point_ids=selected_point_ids,
            cluster_summary=json.dumps(cluster_summary),
            n_points=n_points,
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Inject few-shot examples as alternating user/assistant turns
        for ex in self.few_shot_examples:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({
                "role": "assistant",
                "content": json.dumps(ex["output"]),
            })

        messages.append({"role": "user", "content": user_text})

        try:
            raw_response = self.llm.chat(messages)
        except Exception as e:
            return self._fallback_response(f"LLM call failed: {e}")

        return self._parse_response(raw_response)

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Robustly extract a JSON object from the LLM response."""
        if not raw:
            return self._fallback_response("Empty LLM response")

        # Strip common markdown fences
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # Find the first balanced { ... } block
        json_obj = self._extract_json_object(cleaned)
        if json_obj is None:
            return self._fallback_response(f"Could not parse JSON from: {raw[:200]}")

        try:
            parsed = json.loads(json_obj)
        except json.JSONDecodeError as e:
            return self._fallback_response(f"JSON parse error: {e}")

        # Make sure all expected keys are present
        return {
            "intent": parsed.get("intent", "vague"),
            "complete": bool(parsed.get("complete", False)),
            "constraint": parsed.get("constraint"),
            "followup_question": parsed.get("followup_question"),
            "confirmation_message": parsed.get("confirmation_message"),
        }

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        """Find the first balanced JSON object in the text."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    @staticmethod
    def _fallback_response(error_msg: str) -> Dict[str, Any]:
        return {
            "intent": "vague",
            "complete": False,
            "constraint": None,
            "followup_question": (
                "I had trouble understanding that. Could you rephrase, "
                "or describe what you'd like to do with the selected points?"
            ),
            "confirmation_message": None,
            "_error": error_msg,
        }
