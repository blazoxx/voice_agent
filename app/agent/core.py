from app.agent.schemas import AgentResponse, ChatMessage
from app.agent.state import ConversationState
from app.providers.llm import LLMProvider


class Agent:
    """Core Bloxi agent with session-based conversation memory."""

    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: str,
        state: ConversationState,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.state = state

    def generate_response(self, message: str, session_id: str) -> AgentResponse:
        """Process a user message and persist the assistant response."""

        self.state.add_message(
            session_id,
            ChatMessage(role="user", content=message),
        )

        conversation = self.state.get_messages(session_id)

        llm_messages = [
            {"role": "system", "content": self.system_prompt},
            *[
                {"role": item.role, "content": item.content}
                for item in conversation
            ],
        ]

        response = self.llm.generate(llm_messages)

        self.state.add_message(
            session_id,
            ChatMessage(role="assistant", content=response),
        )

        return AgentResponse(
            response=response,
            session_id=session_id,
        )
