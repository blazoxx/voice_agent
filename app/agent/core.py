import json
from typing import Any

from app.agent.schemas import AgentResponse, ChatMessage, ToolCall
from app.agent.state import ConversationState
from app.agent.tools import ToolRegistry
from app.providers.llm import LLMProvider


class Agent:
    """Bloxi agent with conversation memory and tool execution."""

    MAX_TOOL_ROUNDS = 5

    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: str,
        state: ConversationState,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.state = state
        self.tools = tools or ToolRegistry()

    def generate_response(self, message: str, session_id: str) -> AgentResponse:
        """Process a user message, execute requested tools, and store the reply."""

        self.state.add_message(
            session_id,
            ChatMessage(role="user", content=message),
        )

        conversation = self.state.get_messages(session_id)
        llm_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *[
                {"role": item.role, "content": item.content}
                for item in conversation
            ],
        ]

        tool_calls_made: list[ToolCall] = []
        definitions = self.tools.get_definitions()

        if not definitions:
            response_text = self.llm.generate(llm_messages)
        else:
            api_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": definition.parameters
                        or {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    },
                }
                for definition in definitions
            ]

            response_text = self._run_tool_loop(
                llm_messages,
                api_tools,
                tool_calls_made,
            )

        self.state.add_message(
            session_id,
            ChatMessage(role="assistant", content=response_text),
        )

        return AgentResponse(
            response=response_text,
            session_id=session_id,
            tool_calls=tool_calls_made,
        )

    def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        api_tools: list[dict[str, Any]],
        tool_calls_made: list[ToolCall],
    ) -> str:
        """Execute model-requested tools and continue until a final response."""

        for _ in range(self.MAX_TOOL_ROUNDS):
            model_message = self.llm.generate_with_tools(messages, api_tools)

            tool_calls = getattr(model_message, "tool_calls", None)
            content = getattr(model_message, "content", None)

            if not tool_calls:
                return content or ""

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": [],
            }

            for requested_call in tool_calls:
                function = requested_call.function
                tool_name = function.name

                try:
                    arguments = json.loads(function.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object.")

                    result = self.tools.execute(tool_name, arguments)
                    success = True
                    error = None
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    result = None
                    success = False
                    error = str(exc)

                tool_calls_made.append(
                    ToolCall(name=tool_name, arguments=arguments if "arguments" in locals() and isinstance(arguments, dict) else {})
                )

                assistant_message["tool_calls"].append(
                    {
                        "id": requested_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": function.arguments,
                        },
                    }
                )

                messages.append(assistant_message) if len(assistant_message["tool_calls"]) == 1 else None
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": requested_call.id,
                        "content": json.dumps(
                            {"success": success, "result": result, "error": error}
                        ),
                    }
                )

            # Ensure the assistant tool-call message precedes its tool results.
            # Rebuild this round's sequence in the correct order.
            result_messages = messages[-len(tool_calls):]
            del messages[-len(tool_calls) - 1:]
            messages.append(assistant_message)
            messages.extend(result_messages)

        return "I couldn't complete the requested tool operations within the allowed limit."
