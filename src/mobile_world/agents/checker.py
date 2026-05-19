"""Screenshot-grounded checker for terminal agent claims."""

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from loguru import logger

from mobile_world.agents.base import BaseAgent
from mobile_world.agents.utils.helpers import pil_to_base64
from mobile_world.runtime.utils.models import JSONAction

DEFAULT_CHECKER_PROMPT = """You are a strict visual state checker for a mobile GUI agent.

You will receive the user's task goal, the current screenshot, and the main agent's latest
prediction/action. Your only job is to judge whether the agent is allowed to stop now.

Rules:
- Do not solve the task.
- Do not trust the agent's self-report as fact.
- Use only visible screenshot evidence and the task goal.
- Return res=true only if the screenshot clearly supports that the task is complete or that
  the answer/termination is appropriate.
- Return res=false if the screenshot contradicts the claim or does not provide enough evidence.

Return exactly one JSON object:
{"res": true or false, "reason": "short explanation"}
"""


@dataclass
class CheckerResult:
    res: bool
    reason: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def parse_checker_response(raw_response: str | None) -> CheckerResult:
    if not raw_response:
        return CheckerResult(
            res=True,
            reason="Checker returned empty response; failing open.",
        )

    text = raw_response.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        data = json.loads(text)
        return CheckerResult(
            res=bool(data.get("res", True)),
            reason=str(data.get("reason", "")).strip() or "No reason provided.",
        )
    except Exception as exc:
        return CheckerResult(
            res=True,
            reason=f"Failed to parse checker response; failing open. Error: {exc}. Raw: {raw_response}",
        )


class Checker(BaseAgent):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        prompt: str = DEFAULT_CHECKER_PROMPT,
        retry_times: int = 2,
        **runtime_conf: Any,
    ):
        super().__init__()
        self.model_name = model_name
        self.prompt = prompt
        self.retry_times = retry_times
        self.runtime_conf = {
            "temperature": 0.0,
            "max_tokens": 512,
        }
        self.runtime_conf.update(runtime_conf)
        self.build_openai_client(base_url=base_url, api_key=api_key or "empty")

    def predict(self, observation: dict[str, Any]) -> tuple[str, JSONAction]:
        raise NotImplementedError("Checker does not implement agent prediction")

    def check_finished(
        self,
        *,
        task_goal: str,
        screenshot: Any,
        prediction: str | None,
        action: dict[str, Any],
        step: int,
    ) -> CheckerResult:
        encoded_screenshot = pil_to_base64(screenshot)
        check_payload = {
            "task_goal": task_goal,
            "step": step,
            "main_agent_prediction": prediction,
            "main_agent_action": action,
        }
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(check_payload, ensure_ascii=False, indent=2),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_screenshot}"},
                    },
                ],
            },
        ]

        try:
            raw_response = self.openai_chat_completions_create(
                model=self.model_name,
                messages=messages,
                retry_times=self.retry_times,
                **self.runtime_conf,
            )
        except Exception as exc:
            logger.exception(f"Checker call failed in step {step}: {exc}")
            return CheckerResult(res=True, reason=f"Checker call failed; failing open. Error: {exc}")

        return parse_checker_response(raw_response)
