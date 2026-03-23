import json


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token."""
    total = sum(len(json.dumps(m, default=str)) for m in messages)
    return total // 4


def maybe_compress(messages: list, threshold: int = 80_000) -> list:
    """
    If estimated tokens exceed threshold, truncate the content of old tool
    results (keeping the most recent 4 messages intact).
    """
    if estimate_tokens(messages) <= threshold:
        return messages

    keep_recent = 4  # Always preserve the last N messages in full
    cutoff = max(0, len(messages) - keep_recent)

    compressed = []
    for i, msg in enumerate(messages):
        if i >= cutoff:
            # Keep recent messages untouched
            compressed.append(msg)
            continue

        if msg["role"] == "user" and isinstance(msg["content"], list):
            new_content = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content_str = str(block.get("content", ""))
                    if len(content_str) > 400:
                        new_content.append({
                            **block,
                            "content": content_str[:300] + "\n...[truncated for context management]",
                        })
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)
            compressed.append({**msg, "content": new_content})
        else:
            compressed.append(msg)

    return compressed
