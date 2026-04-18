from __future__ import annotations

import asyncio
import http.client
import http.server
import json
import threading
from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    prompt: str


def make_messages(b: Blackboard) -> list[tinytasktree.JSON]:
    return [{"role": "user", "content": b.prompt}]


def _serve(trace_dir: str) -> tuple[http.server.ThreadingHTTPServer, threading.Thread]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), tinytasktree.create_http_app(trace_dir))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request(server: http.server.ThreadingHTTPServer, method: str, path: str, body: bytes | None = None) -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection(server.server_address[0], server.server_address[1], timeout=5)
    headers = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
    conn.request(method, path, body=body, headers=headers)
    return conn.getresponse()


def test_httpserver_trace_endpoint(tmp_path):
    storage = tinytasktree.FileTraceStorageHandler(str(tmp_path))

    # fmt: off
    tree = (
        tinytasktree.Tree("TraceTree")
        .Sequence()
        ._().Function(lambda: "ok")
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()

    async def _save_trace() -> str:
        async with context.using_blackboard(object()):
            result = await tree(context)
        assert result.is_ok()
        return await storage.save(context.trace_root())

    trace_id = asyncio.run(_save_trace())
    server, thread = _serve(str(tmp_path))
    try:
        resp = _request(server, "GET", f"/trace/{trace_id}")
        data = json.loads(resp.read().decode())
        assert resp.status == 200
        assert data["name"] == "ROOT"
        assert data["kind"] == "ROOT"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_httpserver_llm_non_stream(mock_openai, tmp_path):
    async def handler(**kwargs):
        return {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    server, thread = _serve(str(tmp_path))
    try:
        body = json.dumps(
            {
                "model": "mock/basic",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
        ).encode()
        resp = _request(server, "POST", "/llm", body=body)
        data = json.loads(resp.read().decode())
        assert resp.status == 200
        assert data == {"output": "hello"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_httpserver_llm_passes_reasoning_kwargs(mock_openai, tmp_path):
    recorded = {}

    async def handler(**kwargs):
        recorded.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    server, thread = _serve(str(tmp_path))
    try:
        body = json.dumps(
            {
                "model": "qwen/qwen3.6-plus",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "reasoning": {"enabled": False},
            }
        ).encode()
        resp = _request(server, "POST", "/llm", body=body)
        data = json.loads(resp.read().decode())
        assert resp.status == 200
        assert data == {"output": "ok"}
        assert "reasoning" not in recorded
        assert recorded["extra_body"] == {"reasoning": {"enabled": False}}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_httpserver_llm_stream(mock_openai, tmp_path):
    async def handler(**kwargs):
        assert kwargs["stream"] is True

        async def gen():
            yield {"choices": [{"delta": {"content": "he"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}}
            yield {
                "choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
            }

        return gen()

    mock_openai(handler=handler)

    server, thread = _serve(str(tmp_path))
    try:
        body = json.dumps(
            {
                "model": "mock/stream",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
        ).encode()
        resp = _request(server, "POST", "/llm", body=body)
        data = resp.read().decode()
        assert resp.status == 200
        assert data == "hello"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
