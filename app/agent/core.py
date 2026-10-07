from app.agent.schemas import AgentResponse, ChatMessage
from app.providers.llm import LLMProvider


class Agent:
    """Core Bloxi agent."""

    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: str,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def generate_response(
        self,
        messages: list[ChatMessage],
        session_id: str,
    ) -> AgentResponse:
        llm_messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            *[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
        ]

        response = self.llm.generate(llm_messages)

        return AgentResponse(
            response=response,
            session_id=session_id,
        )