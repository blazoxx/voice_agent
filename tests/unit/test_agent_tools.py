import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.core import Agent
from app.agent.schemas import ChatMessage
from app.agent.state import ConversationState
from app.agent.tools import ToolDefinition, ToolRegistry


def make_tool_call(
    name: str,
    arguments: str,
    call_id: str = "call_1",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def make_model_message(
    content: str | None = None,
    tool_calls: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class FakeToolLLM:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses
        self.tool_messages: list[list[dict[str, Any]]] = []
        self.generate_called = False

    def generate(self, messages: list[dict[str, Any]]) -> str:
        self.generate_called = True
        return "Plain response"

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        self.tool_messages.append([*messages])
        if not self.responses:
            raise AssertionError("Unexpected additional LLM call.")
        return self.responses.pop(0)


from app.tools.calculator_tool import register_calculator

def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_calculator(registry)
    return registry


def make_agent(
    llm: FakeToolLLM,
    registry: ToolRegistry | None = None,
    state: ConversationState | None = None,
) -> Agent:
    return Agent(
        llm=llm,
        system_prompt="You are Bloxi.",
        state=state or ConversationState(),
        tools=registry if registry is not None else make_registry(),
    )


def test_agent_executes_tool_and_returns_final_response() -> None:
    llm = FakeToolLLM(
        [
            make_model_message(
                tool_calls=[
                    make_tool_call(
                        "calculator",
                        '{"expression": "2 + 3"}',
                    )
                ]
            ),
            make_model_message(content="The result is 5."),
        ]
    )
    state = ConversationState()
    agent = make_agent(llm, state=state)

    result = agent.generate_response("What is 2 + 3?", "session_1")

    assert result.response == "The result is 5."
    assert result.session_id == "session_1"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "calculator"
    assert result.tool_calls[0].arguments == {"expression": "2 + 3"}

    second_round = llm.tool_messages[1]
    assistant_tool_message = next(
        message
        for message in second_round
        if message["role"] == "assistant"
        and message.get("tool_calls")
    )
    tool_result_message = next(
        message for message in second_round if message["role"] == "tool"
    )

    assert assistant_tool_message["tool_calls"][0]["id"] == "call_1"
    assert json.loads(tool_result_message["content"])["result"] == 5

    history = state.get_messages("session_1")
    assert [message.role for message in history] == ["user", "assistant"]
    assert history[-1].content == "The result is 5."


def test_invalid_tool_arguments_are_returned_as_tool_errors() -> None:
    llm = FakeToolLLM(
        [
            make_model_message(
                tool_calls=[
                    make_tool_call("calculator", "{invalid json")
                ]
            ),
            make_model_message(content="I couldn't parse the calculation."),
        ]
    )
    agent = make_agent(llm)

    result = agent.generate_response("Calculate something.", "session_2")

    assert result.response == "I couldn't parse the calculation."
    assert result.tool_calls[0].arguments == {}

    tool_result = next(
        message
        for message in llm.tool_messages[1]
        if message["role"] == "tool"
    )
    payload = json.loads(tool_result["content"])
    assert payload["success"] is False
    assert payload["error"]


def test_unknown_tool_is_returned_as_tool_error() -> None:
    llm = FakeToolLLM(
        [
            make_model_message(
                tool_calls=[
                    make_tool_call("missing_tool", "{}")
                ]
            ),
            make_model_message(content="That tool is unavailable."),
        ]
    )
    agent = make_agent(llm)

    result = agent.generate_response("Use the missing tool.", "session_3")

    assert result.response == "That tool is unavailable."

    tool_result = next(
        message
        for message in llm.tool_messages[1]
        if message["role"] == "tool"
    )
    payload = json.loads(tool_result["content"])
    assert payload["success"] is False
    assert "Unknown tool" in payload["error"]


def test_tool_loop_stops_at_maximum_rounds() -> None:
    llm = FakeToolLLM(
        [
            make_model_message(
                tool_calls=[
                    make_tool_call(
                        "calculator",
                        '{"expression": "1 + 1"}',
                        call_id=f"call_{index}",
                    )
                ]
            )
            for index in range(5)
        ]
    )
    agent = make_agent(llm)

    result = agent.generate_response("Keep calculating.", "session_4")

    assert len(llm.tool_messages) == agent.MAX_TOOL_ROUNDS
    assert "allowed limit" in result.response


def test_agent_without_registered_tools_uses_generate() -> None:
    llm = FakeToolLLM([])
    agent = make_agent(llm, registry=ToolRegistry())

    result = agent.generate_response("Hello.", "session_5")

    assert result.response == "Plain response"
    assert llm.generate_called is True
    assert llm.tool_messages == []


def test_multiple_tool_calls_are_ordered_correctly() -> None:
    llm = FakeToolLLM(
        [
            make_model_message(
                tool_calls=[
                    make_tool_call(
                        "calculator",
                        '{"expression": "2 + 3"}',
                        "call_1",
                    ),
                    make_tool_call(
                        "calculator",
                        '{"expression": "4 * 5"}',
                        "call_2",
                    ),
                ]
            ),
            make_model_message(content="Both calculations are complete."),
        ]
    )
    agent = make_agent(llm)

    result = agent.generate_response("Calculate both expressions.", "session_6")

    assert result.response == "Both calculations are complete."

    second_round = llm.tool_messages[1]
    assistant_index = next(
        index
        for index, message in enumerate(second_round)
        if message["role"] == "assistant" and message.get("tool_calls")
    )
    tool_indices = [
        index
        for index, message in enumerate(second_round)
        if message["role"] == "tool"
    ]

    assert len(tool_indices) == 2
    assert assistant_index < tool_indices[0] < tool_indices[1]

    results = [
        json.loads(second_round[index]["content"])["result"]
        for index in tool_indices
    ]
    assert results == [5, 20]
    assert len(result.tool_calls) == 2