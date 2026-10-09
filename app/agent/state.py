from app.agent.schemas import ChatMessage


class ConversationState:
    """Stores conversation history for active sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[ChatMessage]] = {}

    def get_messages(self, session_id: str) -> list[ChatMessage]:
        """Return the conversation history for a session."""
        return self._sessions.get(session_id, []).copy()

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Add a message to a session's conversation history."""
        self._sessions.setdefault(session_id, []).append(message)

    def clear(self, session_id: str) -> None:
        """Clear the conversation history for a session."""
        self._sessions.pop(session_id, None)