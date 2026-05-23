"""POST /chat/stream — SSE chat endpoint."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..agent_stream import agent_event_stream
from ...llm.openai_compatible import LLMClient

router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = []
    vault_path: Optional[str] = None


async def _event_generator(body: ChatRequest, vault_path: str):
    """Async generator that yields SSE-formatted bytes."""
    api_key = os.getenv("PD_API_KEY", "")
    base_url = os.getenv("PD_BASE_URL", "")
    model = os.getenv("PD_MODEL", "")

    if not api_key or not base_url or not model:
        payload = json.dumps({"type": "error", "message": "LLM env not set: PD_API_KEY / PD_BASE_URL / PD_MODEL"})
        yield f"data: {payload}\n\n"
        return

    llm = LLMClient(api_key=api_key, base_url=base_url, model=model)

    async for event in agent_event_stream(
        message=body.message,
        history=body.history,
        vault_path=vault_path,
        llm=llm,
    ):
        yield f"data: {json.dumps(event)}\n\n"


@router.post("/stream")
async def chat_stream(body: ChatRequest, request: Request):
    """Stream agent responses as Server-Sent Events."""
    vault_path = body.vault_path or getattr(request.app.state, "vault_path", "")

    return StreamingResponse(
        _event_generator(body, vault_path),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
