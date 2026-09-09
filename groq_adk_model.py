"""Direct Groq model adapter for Google ADK, without LiteLLM."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, TYPE_CHECKING

from groq import AsyncGroq
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest


def _part_to_message(part: types.Part) -> dict[str, Any] | None:
    if part.text is not None:
        return {"type": "text", "text": part.text}
    if part.function_call is not None:
        call = part.function_call
        return {
            "type": "function",
            "function": {
                "name": call.name or "",
                "arguments": json.dumps(call.args or {}),
            },
        }
    if part.function_response is not None:
        response = part.function_response
        return {
            "role": "tool",
            "tool_call_id": response.id or "",
            "name": response.name or "",
            "content": json.dumps(response.response or {}),
        }
    return None


def _messages(request: LlmRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system_instruction = request.config.system_instruction
    if system_instruction:
        text = (
            system_instruction
            if isinstance(system_instruction, str)
            else " ".join(part.text or "" for part in system_instruction.parts or [])
        )
        messages.append({"role": "system", "content": text})

    for content in request.contents:
        role = "assistant" if content.role in {"model", "assistant"} else "user"
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in content.parts or []:
            if part.function_response is not None:
                messages.append(_part_to_message(part) or {})
            elif part.function_call is not None:
                call = part.function_call
                tool_calls.append(
                    {
                        "id": call.id or "",
                        "type": "function",
                        "function": {
                            "name": call.name or "",
                            "arguments": json.dumps(call.args or {}),
                        },
                    }
                )
            elif part.text:
                text_parts.append(part.text)
        if text_parts or tool_calls:
            message: dict[str, Any] = {"role": role, "content": " ".join(text_parts) or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
    return messages


def _normalize_schema(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {
            key: _normalize_schema(item)
            for key, item in value.items()
        }
        if isinstance(normalized.get("type"), str):
            normalized["type"] = normalized["type"].lower()
        return normalized
    if isinstance(value, list):
        return [_normalize_schema(item) for item in value]
    return value


def _tools(request: LlmRequest) -> list[dict[str, Any]]:
    declarations = []
    for tool in request.config.tools or []:
        declarations.extend(tool.function_declarations or [])
    return [
        {
            "type": "function",
            "function": {
                "name": declaration.name,
                "description": declaration.description or "",
                "parameters": _normalize_schema(
                    declaration.parameters.model_dump(exclude_none=True)
                    if declaration.parameters
                    else {"type": "object", "properties": {}}
                ),
            },
        }
        for declaration in declarations
    ]


class GroqLlm(BaseLlm):
    """ADK BaseLlm implementation that calls GroqCloud directly."""

    _client: AsyncGroq = PrivateAttr()

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        super().__init__(model=model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), **kwargs)
        self._client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

    @property
    def capabilities(self):
        from google.adk.models._capabilities import LlmCapabilities

        return LlmCapabilities(output_schema_and_tools=True)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _messages(llm_request),
            "temperature": 0.2,
        }
        tools = _tools(llm_request)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        parts: list[types.Part] = []
        if message.content:
            parts.append(types.Part(text=message.content))
        for call in message.tool_calls or []:
            arguments = json.loads(call.function.arguments or "{}")
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=call.id,
                        name=call.function.name,
                        args=arguments,
                    )
                )
            )
        finish_reason = None
        if not message.tool_calls:
            finish_reason = types.FinishReason.STOP
            if response.choices[0].finish_reason == "length":
                finish_reason = types.FinishReason.MAX_TOKENS
        usage = response.usage
        usage_metadata = None
        if usage:
            usage_metadata = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=usage.prompt_tokens,
                candidates_token_count=usage.completion_tokens,
                total_token_count=usage.total_tokens,
            )
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            model_version=response.model,
            finish_reason=finish_reason,
            usage_metadata=usage_metadata,
        )
