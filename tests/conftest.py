from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path

import pytest
import redis.asyncio as async_redis

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tinytasktree  # noqa: E402


@pytest.fixture
def mock_litellm(monkeypatch):
    state = {
        "content": '{"greeting": "hello", "numbers": [1, 2, 3]}',
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "hidden": {"response_cost": 0.0},
        "handler": None,
    }

    async def fake_acompletion(**kwargs):
        handler = state["handler"]
        if handler is not None:
            if inspect.iscoroutinefunction(handler):
                return await handler(**kwargs)
            result = handler(**kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        content = state["content"]
        return {
            "choices": [{"message": {"content": content}, "finish_reason": state["finish_reason"]}],
            "usage": state["usage"],
            "_hidden_params": state["hidden"],
        }

    def configure(
        *,
        content: str | None = None,
        finish_reason: str | None = None,
        usage: dict | None = None,
        hidden: dict | None = None,
        handler=None,
    ) -> None:
        if content is not None:
            state["content"] = content
        if finish_reason is not None:
            state["finish_reason"] = finish_reason
        if usage is not None:
            state["usage"] = usage
        if hidden is not None:
            state["hidden"] = hidden
        if handler is not None:
            state["handler"] = handler

    monkeypatch.setattr(tinytasktree.litellm, "acompletion", fake_acompletion)
    return configure


@pytest.fixture
def redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://127.0.0.1:6379")


@pytest.fixture
async def redis_client(redis_url: str):
    client = async_redis.Redis.from_url(redis_url)
    try:
        yield client
    finally:
        await client.aclose()
