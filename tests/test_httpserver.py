"""HTTP server UI and trace route tests."""

from __future__ import annotations

import asyncio
import http.client
import http.server
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import tinytasktree


def _serve(handler: type[http.server.BaseHTTPRequestHandler]):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request(server: http.server.ThreadingHTTPServer, method: str, path: str) -> tuple[int, str, bytes]:
    conn = http.client.HTTPConnection(server.server_address[0], server.server_address[1], timeout=5)
    try:
        conn.request(method, path)
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, resp.getheader("Content-Type", ""), body
    finally:
        conn.close()


def test_httpserver_serves_ui_root_assets_and_spa_fallback(monkeypatch, tmp_path: Path):
    ui_root = tmp_path / "ui_dist"
    assets_dir = ui_root / "assets"
    assets_dir.mkdir(parents=True)
    (ui_root / "index.html").write_text("<html><body>tinytasktree ui</body></html>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('ui');", encoding="utf-8")

    monkeypatch.setattr(tinytasktree, "_find_bundled_ui_root", lambda: ui_root)

    handler = tinytasktree.create_http_app(str(tmp_path / "traces"))
    server, thread = _serve(handler)
    try:
        status, content_type, body = _request(server, "GET", "/")
        assert status == 200
        assert content_type.startswith("text/html")
        assert b"tinytasktree ui" in body

        status, content_type, body = _request(server, "GET", "/trace-demo-id")
        assert status == 200
        assert content_type.startswith("text/html")
        assert b"tinytasktree ui" in body

        status, content_type, body = _request(server, "GET", "/assets/app.js")
        assert status == 200
        assert "javascript" in content_type or "text/plain" in content_type
        assert body == b"console.log('ui');"

        status, content_type, body = _request(server, "GET", "/assets/missing.js")
        assert status == 404
        assert content_type.startswith("application/json")
        assert json.loads(body.decode())["detail"] == "not found: /assets/missing.js"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@dataclass
class Blackboard:
    value: str = ""


def test_httpserver_trace_route_still_wins_over_ui_fallback(monkeypatch, tmp_path: Path):
    ui_root = tmp_path / "ui_dist"
    ui_root.mkdir(parents=True)
    (ui_root / "index.html").write_text("<html><body>tinytasktree ui</body></html>", encoding="utf-8")

    traces_dir = tmp_path / "traces"
    monkeypatch.setattr(tinytasktree, "_find_bundled_ui_root", lambda: ui_root)

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("TraceHttp")
        .Sequence()
        ._().Function(lambda: {"ok": True})
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    async def _run_tree_and_save_trace() -> str:
        async with context.using_blackboard(Blackboard()):
            result = await tree(context)
        assert result.is_ok()
        storage = tinytasktree.FileTraceStorageHandler(str(traces_dir))
        return await storage.save(context.trace_root())

    trace_id = asyncio.run(_run_tree_and_save_trace())

    handler = tinytasktree.create_http_app(str(traces_dir))
    server, thread = _serve(handler)
    try:
        status, content_type, body = _request(server, "GET", f"/trace/{trace_id}")
        assert status == 200
        assert content_type.startswith("application/json")
        payload = json.loads(body.decode())
        assert payload["name"] == "ROOT"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
