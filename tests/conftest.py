from __future__ import annotations

import asyncio
import inspect
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tinytasktree  # noqa: E402


@pytest.fixture
def mock_openai(monkeypatch):
    state = {
        "content": '{"greeting": "hello", "numbers": [1, 2, 3]}',
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "hidden": {"response_cost": 0.0},
        "handler": None,
        "client_kwargs": [],
        "request_kwargs": [],
    }

    class FakeChatCompletions:
        def __init__(self, client_kwargs: dict):
            self._client_kwargs = client_kwargs

        async def create(self, **kwargs):
            state["request_kwargs"].append(kwargs)
            handler = state["handler"]
            if handler is not None:
                if inspect.iscoroutinefunction(handler):
                    return await handler(client_kwargs=self._client_kwargs, **kwargs)
                result = handler(client_kwargs=self._client_kwargs, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result

            content = state["content"]
            return {
                "choices": [{"message": {"content": content}, "finish_reason": state["finish_reason"]}],
                "usage": state["usage"],
                "_hidden_params": state["hidden"],
            }

    class FakeChat:
        def __init__(self, client_kwargs: dict):
            self.completions = FakeChatCompletions(client_kwargs)

    class FakeClient:
        def __init__(self, **kwargs):
            state["client_kwargs"].append(kwargs)
            self.chat = FakeChat(kwargs)

        async def close(self) -> None:
            return None

    def fake_new_async_openai_client(**kwargs):
        return FakeClient(**kwargs)

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

    configure.state = state  # type: ignore[attr-defined]

    monkeypatch.setattr(tinytasktree, "_new_async_openai_client", fake_new_async_openai_client)
    return configure


@pytest.fixture
def memory_store():
    class InMemoryKVStore:
        def __init__(self) -> None:
            self._values: dict[str, tuple[Any, float | None]] = {}

        def _purge_if_expired(self, key: str) -> None:
            entry = self._values.get(key)
            if entry is None:
                return
            _, expires_at = entry
            if expires_at is not None and expires_at <= time.monotonic():
                self._values.pop(key, None)

        async def get(self, key: str) -> Any:
            self._purge_if_expired(key)
            entry = self._values.get(key)
            if entry is None:
                return None
            return entry[0]

        async def set(self, key: str, value: Any, ex: int | float | timedelta | None = None) -> bool:
            expires_at = None
            if ex is not None:
                ttl = ex.total_seconds() if hasattr(ex, "total_seconds") else float(ex)
                expires_at = time.monotonic() + ttl
            self._values[key] = (value, expires_at)
            return True

        async def delete(self, key: str) -> int:
            self._purge_if_expired(key)
            existed = 1 if key in self._values else 0
            self._values.pop(key, None)
            return existed

        async def exists(self, key: str) -> int:
            self._purge_if_expired(key)
            return 1 if key in self._values else 0

    return InMemoryKVStore()
