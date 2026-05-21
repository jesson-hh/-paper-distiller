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


@dataclass
class StreamChunk:
    """A single chunk in a streamed tool-calling response.

    A streamed response is a sequence of these chunks. Text content arrives via
    text_delta; tool_call info arrives across multiple chunks — the first chunk
    for a given call carries tool_call_id, then subsequent chunks fill in
    tool_name_delta and tool_arg_delta. The final chunk's finish_reason is set.
    """

    text_delta: str = ""
    tool_call_id: str | None = None
    tool_name_delta: str = ""
    tool_arg_delta: str = ""
    finish_reason: str = ""


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 600.0,
    ):
        # v1.7: 600s default — deep distillation produces 3-6k Chinese chars
        # of output which qwen3.5-plus can take 4-6 minutes for. Old 120s
        # default caused ReadTimeout mid-call. Override via PD_LLM_TIMEOUT.
        import os
        env_to = os.getenv("PD_LLM_TIMEOUT")
        if env_to:
            try:
                timeout = float(env_to)
            except ValueError:
                pass
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

    def complete_with_tools_stream(
        self,
        messages: list,
        tools: list,
        temperature: float = 0.5,
    ):
        """Stream a tool-calling completion. Yields StreamChunk objects.

        Aliyun Bailian (and other OpenAI-compatible providers) emit SSE lines
        prefixed `data: ` containing JSON. The terminating sentinel is
        `data: [DONE]`. Token usage may arrive in the final non-DONE event.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "stream": True,
            # OpenAI-compatible providers omit `usage` from streamed responses
            # by default. Explicit opt-in lets us track tokens (and hence cost)
            # accurately. Aliyun Bailian and DeepSeek both honor this flag.
            "stream_options": {"include_usage": True},
        }
        seen_call_ids: dict[int, str] = {}
        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if "usage" in data and data["usage"]:
                        self.total_tokens_in += data["usage"].get(
                            "prompt_tokens", 0
                        )
                        self.total_tokens_out += data["usage"].get(
                            "completion_tokens", 0
                        )
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    finish_reason = choice.get("finish_reason") or ""

                    text = delta.get("content") or ""
                    if text:
                        yield StreamChunk(text_delta=text)

                    for tc_delta in delta.get("tool_calls") or []:
                        idx = tc_delta.get("index", 0)
                        new_id: str | None = None
                        if "id" in tc_delta and tc_delta["id"]:
                            if idx not in seen_call_ids:
                                seen_call_ids[idx] = tc_delta["id"]
                                new_id = tc_delta["id"]
                        fn = tc_delta.get("function") or {}
                        name_delta = fn.get("name") or ""
                        arg_delta = fn.get("arguments") or ""
                        if new_id or name_delta or arg_delta:
                            yield StreamChunk(
                                tool_call_id=new_id,
                                tool_name_delta=name_delta,
                                tool_arg_delta=arg_delta,
                            )

                    if finish_reason:
                        yield StreamChunk(finish_reason=finish_reason)
        except httpx.HTTPError as e:
            raise LLMError(f"LLM stream failed: {e}") from e

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
