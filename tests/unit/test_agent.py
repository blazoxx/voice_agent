from app.agent.core import Agent
from app.agent.schemas import ChatMessage


class FakeLLM:
    def generate(self, messages: list[dict[str, str]]) -> str:
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        return "Hello from Bloxi."


def test_agent_generates_response():
    agent = Agent(
        llm=FakeLLM(),
        system_prompt="You are Bloxi.",
    )

    response = agent.generate_response(
        messages=[
            ChatMessage(
                role="user",
                content="Hello",
            )
        ],
        session_id="test-session",
    )

    assert response.response == "Hello from Bloxi."
    assert response.session_id == "test-session"
    assert response.tool_calls == []