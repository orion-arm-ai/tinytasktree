"""Trace storage handler tests.

Steps:
- Save a trace root to disk using FileTraceStorageHandler.
- Query the trace back and verify basic structure.
Expectations:
- Saved trace can be loaded and contains root metadata.
"""

from __future__ import annotations

import tempfile

import tinytasktree


async def test_file_trace_storage_handler_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        handler = tinytasktree.FileTraceStorageHandler(tmpdir)

        # fmt: off
        tree = (
            tinytasktree.Tree("TraceTree")
            .Sequence()
            ._().Function(lambda: "ok")
            .End()
        )
        # fmt: on

        context = tinytasktree.Context()
        async with context.using_blackboard(object()):
            result = await tree(context)

        assert result.is_ok()

        trace_id = await handler.save(context.trace_root())
        loaded = await handler.query(trace_id)

        assert loaded["name"] == "ROOT"
        assert loaded["kind"] == "ROOT"
        assert "children" in loaded
