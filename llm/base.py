from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base for LLM backends (OpenAI-compatible, Anthropic, etc.)."""

    @abstractmethod
    def chat(self, system: str, messages: list, tools: list = None, max_tokens: int = 4096) -> dict:
        """Non-streaming call.

        Returns:
            {
                "content_blocks": [
                    {"type": "text", "text": "..."},
                    {"type": "tool_use", "id": "...", "name": "...", "input": {...}},
                ],
                "stop_reason": "end_turn" | "tool_use"
            }
        """

    @abstractmethod
    def stream_chat(self, system: str, messages: list, tools: list = None,
                     max_tokens: int = 4096, result_holder: dict = None):
        """Streaming call — generator that yields partial text strings.

        After the generator is exhausted, result_holder is populated with:
            result_holder["blocks"]      — list of content blocks (same format as chat())
            result_holder["stop_reason"] — "end_turn" or "tool_use"
        """
