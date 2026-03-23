import json


class ConversationManager:
    """Manages conversation history in Anthropic API format."""

    def __init__(self, gradio_history: list):
        self.messages = []
        # Convert Gradio OpenAI-style history to Anthropic format
        for msg in gradio_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                self.messages.append({"role": role, "content": content})

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, content_blocks: list):
        """Add assistant message from content block dicts.

        Accepts: [{"type":"text","text":"..."}, {"type":"tool_use","id":"...","name":"...","input":{}}]
        """
        self.messages.append({"role": "assistant", "content": list(content_blocks)})

    def add_tool_results(self, results: list):
        """Add tool results as a user message."""
        self.messages.append({"role": "user", "content": results})

    def to_gradio_history(self) -> list:
        """Convert to Gradio's OpenAI-style message list for display."""
        history = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                if content.strip():
                    history.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Extract text blocks; skip tool_use and tool_result blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text" and block.get("text", "").strip():
                            text_parts.append(block["text"])
                if text_parts:
                    history.append({"role": role, "content": "\n".join(text_parts)})

        return history
