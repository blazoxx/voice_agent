from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.agent.core import Agent
from app.agent.schemas import ChatRequest, AgentResponse
from app.agent.state import ConversationState
from app.agent.prompts import BLOXI_SYSTEM_PROMPT
from app.providers.llm import GroqProvider

router = APIRouter()

# Shared instances for this development version.
# Conversation history is in-memory and resets when the process restarts.
conversation_state = ConversationState()
llm_provider = GroqProvider()
agent = Agent(
    llm=llm_provider,
    system_prompt=BLOXI_SYSTEM_PROMPT,
    state=conversation_state,
)


@router.post("/chat", response_model=AgentResponse)
def chat(request: ChatRequest) -> AgentResponse:
    """Process a chat message through Bloxi's Agent Core."""

    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message must not be empty or whitespace-only.",
        )

    session_id = request.session_id or str(uuid4())

    try:
        return agent.generate_response(
            message=message,
            session_id=session_id,
        )
    except Exception as exc:
        # Avoid exposing provider credentials or internal details to clients.
        raise HTTPException(
            status_code=502,
            detail="Bloxi could not generate a response. Please try again.",
        ) from exc
