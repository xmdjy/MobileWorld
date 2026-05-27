from mobile_world.agents.checker import (
    Checker,
    DEFAULT_CHECKER_PROMPT,
    CheckerResult,
    parse_checker_response,
)
from mobile_world.core.runner import (
    _drop_last_agent_prediction_if_supported,
    _execute_single_task,
    _should_check_terminal_action,
)
from mobile_world.runtime.utils.models import ANSWER, FINISHED, JSONAction, Observation


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


def test_should_check_terminal_action_for_finished_and_answer_by_default():
    assert _should_check_terminal_action(JSONAction(action_type=FINISHED))
    assert _should_check_terminal_action(JSONAction(action_type=ANSWER, text="done"))
    assert not _should_check_terminal_action(JSONAction(action_type="unknown"))
    assert not _should_check_terminal_action(JSONAction(action_type="click", x=1, y=2))


def test_should_check_terminal_action_can_disable_answer_trigger():
    assert not _should_check_terminal_action(
        JSONAction(action_type=ANSWER, text="done"),
        enable_answer_trigger=False,
    )


def test_runner_does_not_define_non_terminal_checker_trigger():
    import mobile_world.core.runner as runner

    assert not hasattr(runner, "_should_check_stuck_loop")


def test_stuck_loop_does_not_trigger_checker_or_mas(monkeypatch):
    from PIL import Image

    import mobile_world.core.runner as runner

    class FakeEnv:
        def __init__(self):
            self.screenshot = Image.new("RGB", (2, 2), "white")
            self.executed = []

        def get_task_goal(self, task_type):
            return "Tap the target."

        def initialize_task(self, task_name):
            return Observation(screenshot=self.screenshot)

        def execute_action(self, action):
            self.executed.append(action.model_dump(exclude_none=True))
            return Observation(screenshot=self.screenshot)

        def get_task_score(self, task_type):
            return 0.0, "not relevant"

        def tear_down_task(self, task_type):
            return None

    class FakeAgent:
        def __init__(self):
            self.candidate_calls = 0

        def initialize(self, task_goal):
            pass

        def predict(self, observation):
            return "click target", JSONAction(action_type="click", x=1, y=1)

        def get_total_token_usage(self):
            return {}

        def done(self):
            pass

        def drop_last_prediction_from_history(self):
            return True

        def propose_candidate(self, **kwargs):
            self.candidate_calls += 1
            return "candidate", {"action_type": "navigate_back"}

    class CountingChecker:
        def __init__(self):
            self.calls = 0

        def check_finished(self, **kwargs):
            self.calls += 1
            return CheckerResult(res=False, reason="blocked")

    class MemoryTrajLogger:
        def __init__(self):
            self.entries = []

        def log_tools(self, tools):
            pass

        def log_traj(self, *args, **kwargs):
            self.entries.append(kwargs)

        def log_score(self, score, reason):
            pass

    monkeypatch.setattr(runner, "compute_dhash", lambda screenshot: [0])
    monkeypatch.setattr(runner, "hamming_distance", lambda prev, cur: 0)
    monkeypatch.setattr(runner, "is_no_change", lambda prev, cur, threshold: True)

    env = FakeEnv()
    agent = FakeAgent()
    checker = CountingChecker()
    traj_logger = MemoryTrajLogger()

    _execute_single_task(
        env,
        agent,
        "FakeTask",
        max_step=3,
        traj_logger=traj_logger,
        checker=checker,
        no_change_k=1,
        mas_enabled=True,
    )

    assert checker.calls == 0
    assert agent.candidate_calls == 0
    assert all(entry["trigger_reason"] is None for entry in traj_logger.entries)


def test_terminal_checker_receives_recent_three_screenshots(monkeypatch):
    from PIL import Image

    import mobile_world.core.runner as runner

    class FakeEnv:
        def __init__(self):
            self.step = 0
            self.screenshots = [
                Image.new("RGB", (2, 2), color)
                for color in ("white", "red", "green", "blue", "black")
            ]

        def get_task_goal(self, task_type):
            return "Finish after checking state."

        def initialize_task(self, task_name):
            return Observation(screenshot=self.screenshots[self.step])

        def execute_action(self, action):
            self.step += 1
            return Observation(screenshot=self.screenshots[self.step])

        def get_task_score(self, task_type):
            return 0.0, "not relevant"

        def tear_down_task(self, task_type):
            return None

    class FinishingAgent:
        def __init__(self):
            self.calls = 0

        def initialize(self, task_goal):
            pass

        def predict(self, observation):
            self.calls += 1
            if self.calls < 5:
                return "click", JSONAction(action_type="click", x=1, y=1)
            return "done", JSONAction(action_type=FINISHED)

        def get_total_token_usage(self):
            return {}

        def done(self):
            pass

        def drop_last_prediction_from_history(self):
            return True

    class RecordingChecker:
        def __init__(self):
            self.recent_steps = None

        def check_finished(self, **kwargs):
            self.recent_steps = kwargs.get("recent_steps")
            return CheckerResult(res=False, reason="blocked")

    class MemoryTrajLogger:
        def log_tools(self, tools):
            pass

        def log_traj(self, *args, **kwargs):
            pass

        def log_score(self, score, reason):
            pass

    monkeypatch.setattr(runner, "compute_dhash", lambda screenshot: [0])

    checker = RecordingChecker()

    _execute_single_task(
        FakeEnv(),
        FinishingAgent(),
        "FakeTask",
        max_step=5,
        traj_logger=MemoryTrajLogger(),
        checker=checker,
    )

    assert [item["step"] for item in checker.recent_steps] == [2, 3, 4]
    assert all(item["screenshot"] is not None for item in checker.recent_steps)


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
