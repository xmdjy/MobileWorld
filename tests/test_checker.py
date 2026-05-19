from mobile_world.agents.checker import CheckerResult, parse_checker_response
from mobile_world.core.runner import (
    _drop_last_agent_prediction_if_supported,
    _should_check_finished_action,
)
from mobile_world.runtime.utils.models import FINISHED, JSONAction


def test_should_check_finished_action_only_for_finished_action():
    assert _should_check_finished_action(JSONAction(action_type=FINISHED))
    assert not _should_check_finished_action(JSONAction(action_type="answer", text="done"))
    assert not _should_check_finished_action(JSONAction(action_type="unknown"))
    assert not _should_check_finished_action(JSONAction(action_type="click", x=1, y=2))


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
