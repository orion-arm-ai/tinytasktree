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


async def test_file_trace_storage_handler_generates_sortable_trace_ids_and_lists_desc():
    with tempfile.TemporaryDirectory() as tmpdir:
        handler = tinytasktree.FileTraceStorageHandler(tmpdir)

        # fmt: off
        tree_a = (
            tinytasktree.Tree("Alpha Demo Tree")
            .Sequence()
            ._().Function(lambda: "ok-a")
            .End()
        )
        tree_b = (
            tinytasktree.Tree("Beta Demo Tree")
            .Sequence()
            ._().Function(lambda: "ok-b")
            .End()
        )
        # fmt: on

        context_a = tinytasktree.Context()
        async with context_a.using_blackboard(object()):
            result_a = await tree_a(context_a)
        assert result_a.is_ok()
        trace_id_a = await handler.save(context_a.trace_root())

        context_b = tinytasktree.Context()
        async with context_b.using_blackboard(object()):
            result_b = await tree_b(context_b)
        assert result_b.is_ok()
        trace_id_b = await handler.save(context_b.trace_root())

        assert trace_id_a.endswith(".json") is False
        assert "-alpha-demo-tree-" in trace_id_a
        assert "-beta-demo-tree-" in trace_id_b

        traces = await handler.list_traces()
        assert [trace["id"] for trace in traces[:2]] == [trace_id_b, trace_id_a]
        assert traces[0]["name"] == "Beta Demo Tree"
        assert isinstance(traces[0]["created_at"], str)
