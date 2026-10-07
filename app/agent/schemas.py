from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    name: str
    success: bool
    result: Any = None
    error: str | None = None


class AgentResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[ToolCall] = Field(default_factory=list)