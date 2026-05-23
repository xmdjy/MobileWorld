from mobile_world.agents.checker import (
    Checker,
    DEFAULT_CHECKER_PROMPT,
    CheckerResult,
    parse_checker_response,
)
from mobile_world.core.runner import (
    _drop_last_agent_prediction_if_supported,
    _should_check_finished_action,
)
from mobile_world.runtime.utils.models import FINISHED, JSONAction


class CapturingChecker(Checker):
    def __init__(self):
        self.model_name = "test-model"
        self.prompt = DEFAULT_CHECKER_PROMPT
        self.retry_times = 1
        self.runtime_conf = {}
        self.messages = None

    def openai_chat_completions_create(self, *, messages, **kwargs):
        self.messages = messages
        return '{"res": false, "reason": "captured"}'


def test_should_check_finished_action_only_for_finished_action():
    assert _should_check_finished_action(JSONAction(action_type=FINISHED))
    assert not _should_check_finished_action(JSONAction(action_type="answer", text="done"))
    assert not _should_check_finished_action(JSONAction(action_type="unknown"))
    assert not _should_check_finished_action(JSONAction(action_type="click", x=1, y=2))


def test_runner_does_not_define_non_terminal_checker_trigger():
    import mobile_world.core.runner as runner

    assert not hasattr(runner, "_should_check_stuck_loop")


def test_parse_checker_response_from_json_code_block():
    raw = """```json
    {"res": false, "reason": "The screenshot does not show the requested state."}
    ```"""

    result = parse_checker_response(raw)

    assert result == CheckerResult(
        res=False,
        reason="The screenshot does not show the requested state.",
    )


def test_parse_checker_response_defaults_to_fail_open_on_invalid_json():
    result = parse_checker_response("not json")

    assert result.res is True
    assert "Failed to parse checker response" in result.reason


def test_checker_prompt_prioritizes_task_goal_checklist():
    assert "The user's task goal is the only source of truth for success" in DEFAULT_CHECKER_PROMPT
    assert "decompose the task goal into atomic requirements" in DEFAULT_CHECKER_PROMPT
    assert "goal_checklist" in DEFAULT_CHECKER_PROMPT
    assert "weekend" in DEFAULT_CHECKER_PROMPT
    assert "previous 3 screenshots" in DEFAULT_CHECKER_PROMPT
    assert "not given the main\nagent's thoughts" in DEFAULT_CHECKER_PROMPT
    assert "stuck/no-progress loop" not in DEFAULT_CHECKER_PROMPT


def test_checker_input_excludes_agent_claims_and_actions():
    from PIL import Image

    checker = CapturingChecker()
    screenshot = Image.new("RGB", (1, 1), "white")

    checker.check_finished(
        task_goal="Reply with the required exact body.",
        screenshot=screenshot,
        prediction="I typed the exact body and sent it.",
        action={"action_type": "finished", "text": "success"},
        step=4,
        recent_steps=[
            {
                "step": 1,
                "prediction": "I typed a message that is not visible.",
                "action": {"action_type": "input_text", "text": "untrusted body"},
                "screenshot": screenshot,
            }
        ],
        trigger_reason="completion_check",
    )

    text_parts = [
        item["text"]
        for message in checker.messages
        for item in message["content"]
        if item["type"] == "text"
    ]
    combined_text = "\n".join(text_parts)

    assert "Reply with the required exact body." in combined_text
    assert "I typed" not in combined_text
    assert "untrusted body" not in combined_text
    assert "main_agent_prediction" not in combined_text
    assert "main_agent_action" not in combined_text


def test_checker_input_keeps_only_last_three_historical_screenshots():
    from PIL import Image

    checker = CapturingChecker()
    screenshot = Image.new("RGB", (1, 1), "white")

    checker.check_finished(
        task_goal="Verify the visible state.",
        screenshot=screenshot,
        prediction=None,
        action={},
        step=5,
        recent_steps=[
            {"step": 1, "screenshot": screenshot},
            {"step": 2, "screenshot": screenshot},
            {"step": 3, "screenshot": screenshot},
            {"step": 4, "screenshot": screenshot},
        ],
    )

    text_parts = [
        item["text"]
        for message in checker.messages
        for item in message["content"]
        if item["type"] == "text"
    ]
    combined_text = "\n".join(text_parts)

    assert "Historical screenshot for step 1" not in combined_text
    assert "Historical screenshot for step 2" in combined_text
    assert "Historical screenshot for step 3" in combined_text
    assert "Historical screenshot for step 4" in combined_text


def test_drop_last_agent_prediction_calls_optional_agent_hook():
    class AgentWithHistoryDrop:
        def __init__(self):
            self.called = False

        def drop_last_prediction_from_history(self):
            self.called = True
            return True

    agent = AgentWithHistoryDrop()

    assert _drop_last_agent_prediction_if_supported(agent) is True
    assert agent.called is True


def test_drop_last_agent_prediction_ignores_agents_without_hook():
    class AgentWithoutHistoryDrop:
        pass

    assert _drop_last_agent_prediction_if_supported(AgentWithoutHistoryDrop()) is False
