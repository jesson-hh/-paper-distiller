"""OpenAI-compatible LLM client using raw HTTP (no SDK dependency).

Works with any OpenAI-compatible endpoint:
- Alibaba Bailian Coding Plan (coding.dashscope.aliyuncs.com)
- DashScope compatible-mode
- OpenRouter, DeepSeek, Ollama, vLLM, etc.
"""

import json
import uuid
import httpx
from .base import LLMClient


def _convert_tools(anthropic_tools: list) -> list:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    openai_tools = []
    for t in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        })
    return openai_tools


def _convert_messages(system: str, messages: list) -> list:
    """Convert Anthropic-style messages to OpenAI chat format."""
    oai_messages = []
    if system:
        oai_messages.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            oai_messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            has_tool_use = any(b.get("type") == "tool_use" for b in content if isinstance(b, dict))
            has_tool_result = any(b.get("type") == "tool_result" for b in content if isinstance(b, dict))

            if role == "assistant" and has_tool_use:
                text_parts = []
                tool_calls = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and block.get("text", "").strip():
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                            },
                        })
                oai_messages.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    "tool_calls": tool_calls,
                })

            elif role == "user" and has_tool_result:
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        oai_messages.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        })
            else:
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                if text_parts:
                    oai_messages.append({"role": role, "content": "\n".join(text_parts)})

    return oai_messages


def _parse_sse_line(line: str):
    """Parse a single SSE data line, return parsed JSON or None."""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return "DONE"
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


class OpenAIClient(LLMClient):
    """LLM client using raw HTTP calls to OpenAI-compatible APIs."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.api_key = api_key
        # Ensure base_url ends without trailing slash
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout = 120.0

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _endpoint(self):
        return f"{self.base_url}/chat/completions"

    def chat(self, system: str, messages: list, tools: list = None, max_tokens: int = 4096) -> dict:
        body = {
            "model": self.model,
            "messages": _convert_messages(system, messages),
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = _convert_tools(tools)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self._endpoint(), json=body, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        return self._parse_response(data)

    def stream_chat(self, system: str, messages: list, tools: list = None,
                     max_tokens: int = 4096, result_holder: dict = None):
        if result_holder is None:
            result_holder = {}

        body = {
            "model": self.model,
            "messages": _convert_messages(system, messages),
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            body["tools"] = _convert_tools(tools)

        current_text = ""
        tool_calls_acc = {}  # index -> {id, name, arguments}
        stop_reason = "end_turn"

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", self._endpoint(), json=body, headers=self._headers()) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    parsed = _parse_sse_line(line)
                    if parsed is None:
                        continue
                    if parsed == "DONE":
                        break

                    choices = parsed.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    # Text content
                    if delta.get("content"):
                        current_text += delta["content"]
                        yield current_text

                    # Tool calls
                    if delta.get("tool_calls"):
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": tc_delta.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc_delta.get("id"):
                                tool_calls_acc[idx]["id"] = tc_delta["id"]
                            func = tc_delta.get("function", {})
                            if func.get("name"):
                                tool_calls_acc[idx]["name"] = func["name"]
                            if func.get("arguments"):
                                tool_calls_acc[idx]["arguments"] += func["arguments"]

                    # Finish reason
                    finish = choice.get("finish_reason")
                    if finish:
                        if finish in ("tool_calls", "function_call"):
                            stop_reason = "tool_use"
                        else:
                            stop_reason = "end_turn"

        # Build content blocks
        content_blocks = []
        if current_text:
            content_blocks.append({"type": "text", "text": current_text})
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            try:
                input_data = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                input_data = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": input_data,
            })

        result_holder["blocks"] = content_blocks
        result_holder["stop_reason"] = stop_reason

    def _parse_response(self, data: dict) -> dict:
        choice = data["choices"][0]
        message = choice["message"]
        content_blocks = []

        if message.get("content"):
            content_blocks.append({"type": "text", "text": message["content"]})

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                try:
                    input_data = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    input_data = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": input_data,
                })

        finish = choice.get("finish_reason", "stop")
        stop_reason = "tool_use" if finish in ("tool_calls", "function_call") else "end_turn"

        return {
            "content_blocks": content_blocks,
            "stop_reason": stop_reason,
        }
