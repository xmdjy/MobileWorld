"""Screenshot-grounded checker for terminal agent decisions."""

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from loguru import logger

from mobile_world.agents.base import BaseAgent
from mobile_world.agents.utils.helpers import pil_to_base64
from mobile_world.runtime.utils.models import JSONAction

DEFAULT_CHECKER_PROMPT = """You are a strict visual verifier for a mobile GUI agent.

You will receive:
- the user's task goal
- the current screenshot
- a short visual trajectory window containing up to the previous 3 screenshots

Your job is to decide whether the agent is allowed to stop now.

Highest-priority principle:
The user's task goal is the only source of truth for success. Treat the task goal as
a mandatory acceptance checklist, not as background context. Before judging the
current state, decompose the task goal into atomic requirements and verify each
requirement one by one against the screenshots.

Use only screenshots as evidence. You are intentionally not given the main
agent's thoughts, claims, attempted action text, or typed text because those are
not reliable evidence of what happened on the device. Historical screenshots are
useful for checking whether recent UI states changed; they do not override the
task goal.

A task is allowed to finish only when the current screenshot, supported by the
short visual trajectory window if needed, provides direct, visible, and
task-specific evidence that every atomic requirement is satisfied. Partial
completion is failure to verify.

Do not infer completion from:
- any agent self-report
- the fact that a related app/page/object is visible
- absence of visible errors
- returning to the home screen or inbox
- seeing the original email/message/item without evidence of the required response or side effect
- seeing a generic success toast unless the screenshot also supports the required recipient/context/content
- typed text, sent content, or settings values that are not visible in any screenshot

Goal decomposition rules:
- Break the task goal into the smallest independently checkable requirements.
- Preserve qualifiers and quantifiers exactly. For example, "weekend" means both
  Saturday and Sunday; it does not mean only Saturday. "maximum" means the control
  is at its highest available value, not merely increased. "off" means visibly disabled.
- Do not collapse multiple requirements into one broad statement.
- Do not mark a requirement verified unless the screenshot directly supports it.
- If any required condition is not visible, ambiguous, or only implied by an
  assumed action or side effect, mark that requirement as "insufficient".
- If any requirement is "contradicted" or "insufficient", set res=false.

Decision rules:
- Return "verified" only if the screenshot directly proves completion of every key requirement in the task goal.
- Return "contradicted" if the screenshot clearly shows that at least one requirement is not satisfied.
- Return "insufficient" if the screenshot is relevant but does not prove completion.
- If uncertain, choose "insufficient".
- Never use "verified" from an assumed action or side effect that is not visible.

For communication tasks such as email/SMS/chat:
- Verify the correct recipient or thread.
- Verify the required message content when visible.
- Verify that the message was actually sent, not merely typed or drafted.
- If only the inbox, home screen, contact list, original email, or compose screen is visible,
  return "insufficient" unless the sent message itself or a context-specific sent confirmation is visible.

For settings tasks:
- Verify each requested setting separately.
- If the task specifies multiple conditions, all must be visible or otherwise directly supported by the screenshot.
- For repeat days, toggles, sliders, ringtone, vibration, brightness, font/display size, etc.,
  do not assume correctness unless the relevant UI state is visible.
- For day selections, list each required day separately in the checklist and verify
  that no forbidden days are selected when the task implies exclusivity.

For file/app/database-like side effects:
- If the screenshot does not show the target file/item/folder/post/event/message in its final expected state,
  return "insufficient".
- Do not assume that a previous click successfully completed the operation.

Return exactly one JSON object with this schema:
{
  "status": "verified" | "contradicted" | "insufficient",
  "res": true or false,
  "trigger_assessment": "completion_check" | "other",
  "goal_checklist": [
    {
      "requirement": "one atomic requirement copied or paraphrased from task_goal",
      "status": "verified" | "contradicted" | "insufficient",
      "evidence": "specific visible evidence, or why the screenshot is insufficient"
    }
  ],
  "invalid_beliefs": ["unsupported assumptions that should not be inferred from screenshots"],
  "recommended_constraint": "short instruction for what the main agent must avoid or verify next",
  "visible_evidence": ["short bullet-like evidence from the screenshot"],
  "missing_evidence": ["required evidence that is not visible or not proven"],
  "reason": "short explanation"
}

Set res=true only when status is "verified".
Set res=false when status is "contradicted" or "insufficient".
The top-level status must be "verified" only if every goal_checklist item is "verified".

Be conservative: false positives are worse than false negatives.
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
            "max_tokens": 1024,
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
        recent_steps: list[dict[str, Any]] | None = None,
        trigger_reason: str = "completion_check",
    ) -> CheckerResult:
        encoded_screenshot = pil_to_base64(screenshot)
        recent_steps = (recent_steps or [])[-3:]
        check_payload = {
            "task_goal": task_goal,
            "step": step,
            "trigger_reason": trigger_reason,
            "visual_input": {
                "historical_screenshots": [
                    {"step": item.get("step")}
                    for item in recent_steps
                    if item.get("screenshot") is not None
                ],
                "current_screenshot": {"step": step},
            },
        }
        user_content = [
            {
                "type": "text",
                "text": json.dumps(check_payload, ensure_ascii=False, indent=2),
            },
        ]
        for item in recent_steps:
            historical_screenshot = item.get("screenshot")
            if historical_screenshot is None:
                continue
            encoded_historical_screenshot = pil_to_base64(historical_screenshot)
            user_content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Historical screenshot for step {item.get('step')}:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_historical_screenshot}"
                        },
                    },
                ]
            )
        user_content.extend(
            [
                {"type": "text", "text": f"Current screenshot for step {step}:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_screenshot}"},
                },
            ]
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.prompt}]},
            {
                "role": "user",
                "content": user_content,
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
