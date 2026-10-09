from app.agent.core import Agent
from app.agent.schemas import ChatMessage
from app.agent.state import ConversationState


class FakeLLM:
    """Deterministic LLM substitute for unit testing."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return f"Reply {len(self.calls)} from Bloxi."


def test_agent_stores_user_and_assistant_messages():
    state = ConversationState()
    llm = FakeLLM()
    agent = Agent(
        llm=llm,
        system_prompt="You are Bloxi.",
        state=state,
    )

    response = agent.generate_response(
        message="Hello",
        session_id="session-1",
    )

    history = state.get_messages("session-1")

    assert response.response == "Reply 1 from Bloxi."
    assert response.session_id == "session-1"
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Hello"
    assert history[1].role == "assistant"
    assert history[1].content == "Reply 1 from Bloxi."


def test_agent_sends_system_prompt_and_conversation_history():
    state = ConversationState()
    llm = FakeLLM()
    agent = Agent(
        llm=llm,
        system_prompt="You are Bloxi.",
        state=state,
    )

    agent.generate_response("Hello", "session-1")
    agent.generate_response("How are you?", "session-1")

    second_call = llm.calls[1]

    assert second_call[0] == {
        "role": "system",
        "content": "You are Bloxi.",
    }
    assert second_call[1] == {
        "role": "user",
        "content": "Hello",
    }
    assert second_call[2] == {
        "role": "assistant",
        "content": "Reply 1 from Bloxi.",
    }
    assert second_call[3] == {
        "role": "user",
        "content": "How are you?",
    }


def test_agent_keeps_sessions_isolated():
    state = ConversationState()
    llm = FakeLLM()
    agent = Agent(
        llm=llm,
        system_prompt="You are Bloxi.",
        state=state,
    )

    agent.generate_response("Message A", "session-A")
    agent.generate_response("Message B", "session-B")

    messages_for_a = llm.calls[0]
    messages_for_b = llm.calls[1]

    assert not any(
        item["content"] == "Message B"
        for item in messages_for_a
    )
    assert not any(
        item["content"] == "Message A"
        for item in messages_for_b
    )
    assert len(state.get_messages("session-A")) == 2
    assert len(state.get_messages("session-B")) == 2
