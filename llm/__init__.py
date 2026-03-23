import os
from .base import LLMClient


def _get_config():
    """Read LLM configuration from env with backward compatibility."""
    api_key = os.environ.get("API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = (os.environ.get("BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL", "")).strip()
    model = os.environ.get("MODEL") or os.environ.get("ANTHROPIC_MODEL", "")
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
    return provider, api_key, base_url, model


def get_client(model_override: str = None) -> LLMClient:
    """Factory: create LLM client based on LLM_PROVIDER env var.

    Args:
        model_override: Use a different model than the default (e.g. for proof_tool).
    """
    provider, api_key, base_url, model = _get_config()
    if model_override:
        model = model_override

    if provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(api_key=api_key, base_url=base_url, model=model)
    else:
        from .openai_client import OpenAIClient
        return OpenAIClient(api_key=api_key, base_url=base_url, model=model)
