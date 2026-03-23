"""Anthropic-compatible LLM client using raw HTTP (no SDK dependency).

Works with Anthropic official API and compatible endpoints.
"""

import json
import httpx
from .base import LLMClient


def _parse_sse_line(line: str):
    """Parse a single SSE line."""
    line = line.strip()
    if not line:
        return None, None
    if line.startswith("event:"):
        return "event", line[len("event:"):].strip()
    if line.startswith("data:"):
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return "data", "DONE"
        try:
            return "data", json.loads(data)
        except json.JSONDecodeError:
            return None, None
    return None, None


class AnthropicClient(LLMClient):
    """LLM client using raw HTTP calls to Anthropic Messages API."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self.timeout = 120.0

    def _headers(self):
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _endpoint(self):
        # Handle both "https://api.anthropic.com" and "https://xxx/apps/anthropic"
        base = self.base_url
        if base.endswith("/anthropic"):
            return f"{base}/v1/messages"
        return f"{base}/v1/messages"

    def chat(self, system: str, messages: list, tools: list = None, max_tokens: int = 4096) -> dict:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self._endpoint(), json=body, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        return self._parse_response(data)

    def stream_chat(self, system: str, messages: list, tools: list = None,
                     max_tokens: int = 4096, result_holder: dict = None):
        if result_holder is None:
            result_holder = {}

        try:
            yield from self._do_stream(system, messages, tools, max_tokens, result_holder)
        except Exception:
            # Fallback to non-streaming
            result = self.chat(system, messages, tools, max_tokens)
            result_holder["blocks"] = result["content_blocks"]
            result_holder["stop_reason"] = result["stop_reason"]
            for block in result["content_blocks"]:
                if block["type"] == "text":
                    yield block["text"]

    def _do_stream(self, system, messages, tools, max_tokens, result_holder):
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": True,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools

        content_blocks = []
        current_text = ""
        current_tool = None
        current_tool_json = ""
        stop_reason = "end_turn"
        current_event = None

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", self._endpoint(), json=body, headers=self._headers()) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    kind, value = _parse_sse_line(line)

                    if kind == "event":
                        current_event = value
                        continue

                    if kind != "data" or value is None:
                        continue
                    if value == "DONE":
                        break

                    data = value
                    etype = data.get("type", current_event or "")

                    if etype == "content_block_start":
                        block = data.get("content_block", {})
                        if block.get("type") == "text":
                            current_text = block.get("text", "")
                        elif block.get("type") == "tool_use":
                            current_tool = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "input": {},
                            }
                            current_tool_json = ""

                    elif etype == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            current_text += delta.get("text", "")
                            yield current_text
                        elif delta.get("type") == "input_json_delta":
                            current_tool_json += delta.get("partial_json", "")

                    elif etype == "content_block_stop":
                        if current_tool is not None:
                            try:
                                current_tool["input"] = json.loads(current_tool_json) if current_tool_json else {}
                            except json.JSONDecodeError:
                                current_tool["input"] = {}
                            content_blocks.append({
                                "type": "tool_use",
                                **current_tool,
                            })
                            current_tool = None
                            current_tool_json = ""
                        elif current_text:
                            content_blocks.append({"type": "text", "text": current_text})
                            current_text = ""

                    elif etype == "message_delta":
                        delta = data.get("delta", {})
                        if "stop_reason" in delta and delta["stop_reason"]:
                            stop_reason = delta["stop_reason"]

                    elif etype == "message_stop":
                        pass

        result_holder["blocks"] = content_blocks
        result_holder["stop_reason"] = stop_reason

    def _parse_response(self, data: dict) -> dict:
        content_blocks = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content_blocks.append({"type": "text", "text": block.get("text", "")})
            elif block.get("type") == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                })
        return {
            "content_blocks": content_blocks,
            "stop_reason": data.get("stop_reason", "end_turn"),
        }
