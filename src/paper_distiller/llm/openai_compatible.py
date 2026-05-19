"""OpenAI-compatible Chat Completions client.

Works with any provider exposing /v1/chat/completions: Aliyun Bailian,
DeepSeek, OpenRouter, local Ollama, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class ToolCallResponse:
    """Response from a function-calling-enabled completion.

    text may be empty if the assistant only emitted tool_calls;
    tool_calls may be empty if the assistant produced a plain text reply.
    """

    text: str = ""
    tool_calls: list = field(default_factory=list)
    finish_reason: str = ""


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self._client = httpx.Client(timeout=timeout)

    @property
    def estimated_cost_cny(self) -> float:
        """Total session cost in CNY based on accumulated in/out tokens."""
        from .pricing import estimate_cost_cny
        return estimate_cost_cny(
            self.model, self.total_tokens_in, self.total_tokens_out
        )

    def complete(
        self,
        messages: list,
        temperature: float = 0.7,
        response_format: str | None = None,
    ) -> str:
        """Send messages to the LLM, return the assistant content string.

        response_format="json" requests strict JSON object output.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format == "json":
            body["response_format"] = {"type": "json_object"}

        try:
            r = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {e}") from e

        data = r.json()
        if "usage" in data:
            self.total_tokens_in += data["usage"].get("prompt_tokens", 0)
            self.total_tokens_out += data["usage"].get("completion_tokens", 0)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected LLM response shape: {data}") from e

    def complete_with_tools(
        self,
        messages: list,
        tools: list,
        temperature: float = 0.5,
    ) -> ToolCallResponse:
        """Complete with OpenAI-style function-calling enabled.

        Returns ToolCallResponse with .text (if assistant replied in natural
        language) and/or .tool_calls (if the assistant chose to invoke tools).
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
        }

        try:
            r = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {e}") from e

        try:
            data = r.json()
        except json.JSONDecodeError as e:
            raise LLMError(f"non-JSON response: {r.text[:200]}") from e

        if "usage" in data:
            self.total_tokens_in += data["usage"].get("prompt_tokens", 0)
            self.total_tokens_out += data["usage"].get("completion_tokens", 0)

        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"unexpected LLM response shape: {data}")
        choice = choices[0]
        msg = choice.get("message", {}) or {}
        text = msg.get("content") or ""

        tool_calls: list = []
        for raw in msg.get("tool_calls") or []:
            fn = raw.get("function", {}) or {}
            try:
                args = json.loads(fn.get("arguments", "{}") or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=raw.get("id", "") or "",
                    name=fn.get("name", "") or "",
                    arguments=args,
                )
            )

        return ToolCallResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "") or "",
        )

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
