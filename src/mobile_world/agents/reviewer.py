"""Trajectory reviewer.

Single-shot LLM module that looks back at the main agent's accumulated
trajectory and emits a short text hint. The hint is injected into the main
agent's prompt at the current step.
"""

import json
import time
from typing import Any

from loguru import logger
from openai import OpenAI

DEFAULT_REVIEWER_PROMPT = """你是 trajectory reviewer。你将看到主线 agent 在当前任务中累积的推理轨迹（thought + action 序列），请阅读并分析其推理过程，输出一段简短的 text 用于辅助主线 agent 完成任务。

主线 agent 是一个 GUI agent，每步只能看到当前一张截图，所以容易：
- 忘记原始 task 目标（goal drift）
- 不验证上一步 action 是否真生效，反复重试无效 action
- 中途观察到的关键事实（数字 / 日期 / 计数 / ID / 验证码）在后续步骤被遗忘
- 推理过程中出现明显矛盾或凭"visible 部分"就下结论

你的职责是从轨迹里 surface 出当前最值得提醒的信息。请用 1-3 句话直接给出 hint，不要赘述分析过程。如果当前没什么特别要提醒的，只输出 "OK"。

{task_goal_section}主线 agent 累积 trajectory:
```json
{traj_text}
```
"""


class Reviewer:
    """Look-back reviewer that emits a short hint based on the accumulated trajectory."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "empty",
        prompt_template: str = DEFAULT_REVIEWER_PROMPT,
        runtime_conf: dict[str, Any] | None = None,
        retry_times: int = 2,
    ):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "empty",
            timeout=60.0,
        )
        self.prompt_template = prompt_template
        self.runtime_conf = runtime_conf or {"temperature": 0.0, "max_tokens": 256}
        self.retry_times = retry_times

    def _build_prompt(self, traj: list[dict[str, Any]] | str, task_goal: str) -> str:
        if isinstance(traj, str):
            traj_text = traj if traj.strip() else "(尚无累积步骤，这是 task 的第一步)"
        elif traj:
            traj_text = json.dumps(traj, ensure_ascii=False, indent=2)
        else:
            traj_text = "(尚无累积步骤，这是 task 的第一步)"

        task_goal_section = (
            f"当前任务目标 (task_goal):\n{task_goal.strip()}\n\n"
            if task_goal and task_goal.strip()
            else ""
        )
        return self.prompt_template.format(
            task_goal_section=task_goal_section,
            traj_text=traj_text,
        )

    def _model_compat_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Apply model-specific kwarg tweaks consistent with BaseAgent.openai_chat_completions_create."""
        kwargs = dict(kwargs)
        model = self.model_name.lower()
        if "claude" in model:
            kwargs["max_tokens"] = max(kwargs.get("max_tokens", 256), 1024)
            kwargs.pop("temperature", None)
        if "gpt" in model or "o1" in model:
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        if "kimi-k" in model:
            kwargs["extra_body"] = {"enable_thinking": True}
        return kwargs

    def review(
        self,
        traj: list[dict[str, Any]] | str,
        task_goal: str = "",
    ) -> str:
        """Return a short hint for the current step.

        Args:
            traj: Structured traj list (list of step dicts) or pre-serialized JSON string.
            task_goal: Original task goal. Always passed so reviewer can surface goal-recall
                hints even on step 1 when traj is still empty.

        Returns:
            Hint text. Empty string when there's no useful input (empty traj AND empty task_goal)
            or when the LLM call fails.
        """
        if not traj and not (task_goal and task_goal.strip()):
            return ""

        prompt = self._build_prompt(traj, task_goal)
        kwargs = self._model_compat_kwargs(self.runtime_conf)

        for attempt in range(self.retry_times + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
                hint = (response.choices[0].message.content or "").strip()
                logger.info(f"[Reviewer] hint: {hint}")
                return hint
            except Exception as e:
                logger.warning(
                    f"[Reviewer] call failed (attempt {attempt + 1}/{self.retry_times + 1}): {e}"
                )
                if attempt < self.retry_times:
                    time.sleep(1)
        logger.warning("[Reviewer] all retries exhausted, returning empty hint")
        return ""
