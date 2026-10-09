from app.agent.schemas import ChatMessage
from app.agent.state import ConversationState


def test_new_session_has_no_messages():
    state = ConversationState()

    messages = state.get_messages("session-1")

    assert messages == []


def test_messages_are_stored_per_session():
    state = ConversationState()

    state.add_message(
        "session-1",
        ChatMessage(role="user", content="Hello"),
    )

    state.add_message(
        "session-1",
        ChatMessage(role="assistant", content="Hello from Bloxi."),
    )

    messages = state.get_messages("session-1")

    assert len(messages) == 2
    assert messages[0].content == "Hello"
    assert messages[1].content == "Hello from Bloxi."


def test_sessions_are_isolated():
    state = ConversationState()

    state.add_message(
        "session-1",
        ChatMessage(role="user", content="Message one"),
    )

    state.add_message(
        "session-2",
        ChatMessage(role="user", content="Message two"),
    )

    assert len(state.get_messages("session-1")) == 1
    assert len(state.get_messages("session-2")) == 1
    assert state.get_messages("session-1")[0].content == "Message one"
    assert state.get_messages("session-2")[0].content == "Message two"


def test_clear_removes_session_history():
    state = ConversationState()

    state.add_message(
        "session-1",
        ChatMessage(role="user", content="Hello"),
    )

    state.clear("session-1")

    assert state.get_messages("session-1") == []


def test_get_messages_returns_copy():
    state = ConversationState()

    state.add_message(
        "session-1",
        ChatMessage(role="user", content="Hello"),
    )

    messages = state.get_messages("session-1")
    messages.clear()

    assert len(state.get_messages("session-1")) == 1