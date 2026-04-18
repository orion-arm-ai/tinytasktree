"""Cacher behavior tests.

Steps:
- Run a Cacher without a value_validator to observe miss then hit.
- Run a Cacher with a value_validator to observe hit when validator matches and miss when it changes.
Expectations:
- Miss calls the child and stores the result.
- Hit returns cached value without running the child.
- Validator change invalidates cache and triggers the child.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    key: str
    value: int = 0
    validator: str = "v1"


def compute(b: Blackboard) -> int:
    b.value += 1
    return b.value


def _find_first_trace_by_kind(root: tinytasktree.TraceNode, kind: str) -> tinytasktree.TraceNode:
    stack = [root]
    while stack:
        node = stack.pop()
        if node.kind == kind:
            return node
        stack.extend(node.children.values())
    raise AssertionError(f"trace node kind not found: {kind}")


async def test_cacher_without_value_validator_hit_and_miss(memory_store):
    key = f"test:cacher:{uuid.uuid4()}"
    await memory_store.delete(key)

    try:
        calls: list[str] = []

        def compute_counted(b: Blackboard) -> int:
            calls.append("call")
            return compute(b)

        # fmt: off
        tree = (
            tinytasktree.Tree[Blackboard]("CacherNoValidator")
            .Cacher(key_func=lambda b: b.key, store=memory_store, expiration=5)
            ._().Function(compute_counted)
            .End()
        )
        # fmt: on

        context = tinytasktree.Context()
        blackboard = Blackboard(key=key, value=0)
        async with context.using_blackboard(blackboard):
            result = await tree(context)
        assert result.is_ok()
        assert result.data == 1
        assert blackboard.value == 1
        assert len(calls) == 1
        trace = _find_first_trace_by_kind(context.trace_root(), "Cacher")
        assert trace.attributes["cache_key"] == key
        assert trace.attributes["cache_enabled"] is True
        assert trace.attributes["cache_hit"] is False
        assert trace.attributes["cache_status"] == "miss"
        assert trace.attributes["cache_written"] is True

        context = tinytasktree.Context()
        blackboard = Blackboard(key=key, value=0)
        async with context.using_blackboard(blackboard):
            result = await tree(context)
        assert result.is_ok()
        assert result.data == 1
        assert blackboard.value == 0
        assert len(calls) == 1
        trace = _find_first_trace_by_kind(context.trace_root(), "Cacher")
        assert trace.attributes["cache_key"] == key
        assert trace.attributes["cache_enabled"] is True
        assert trace.attributes["cache_hit"] is True
        assert trace.attributes["cache_status"] == "hit"
        assert any("cache hit" in log for log in trace.logs)
    finally:
        await memory_store.delete(key)


async def test_cacher_with_value_validator_hit_and_miss(memory_store):
    key = f"test:cacher:{uuid.uuid4()}"
    await memory_store.delete(key)

    try:
        calls: list[str] = []

        def compute_counted(b: Blackboard) -> int:
            calls.append("call")
            return compute(b)

        # fmt: off
        tree = (
            tinytasktree.Tree[Blackboard]("CacherWithValidator")
            .Cacher(
                key_func=lambda b: b.key,
                store=memory_store,
                expiration=5,
                value_validator=lambda b: b.validator,
            )
            ._().Function(compute_counted)
            .End()
        )
        # fmt: on

        context = tinytasktree.Context()
        blackboard = Blackboard(key=key, value=0, validator="v1")
        async with context.using_blackboard(blackboard):
            result = await tree(context)
        assert result.is_ok()
        assert result.data == 1
        assert blackboard.value == 1
        assert len(calls) == 1
        trace = _find_first_trace_by_kind(context.trace_root(), "Cacher")
        assert trace.attributes["cache_key"] == key
        assert trace.attributes["cache_validation"] == "v1"
        assert trace.attributes["cache_hit"] is False
        assert trace.attributes["cache_status"] == "miss"

        context = tinytasktree.Context()
        blackboard = Blackboard(key=key, value=0, validator="v1")
        async with context.using_blackboard(blackboard):
            result = await tree(context)
        assert result.is_ok()
        assert result.data == 1
        assert blackboard.value == 0
        assert len(calls) == 1
        trace = _find_first_trace_by_kind(context.trace_root(), "Cacher")
        assert trace.attributes["cache_validation"] == "v1"
        assert trace.attributes["cache_hit"] is True
        assert trace.attributes["cache_status"] == "hit"
        assert any("cache hit" in log for log in trace.logs)

        context = tinytasktree.Context()
        blackboard = Blackboard(key=key, value=0, validator="v2")
        async with context.using_blackboard(blackboard):
            result = await tree(context)
        assert result.is_ok()
        assert result.data == 1
        assert blackboard.value == 1
        assert len(calls) == 2
        trace = _find_first_trace_by_kind(context.trace_root(), "Cacher")
        assert trace.attributes["cache_validation"] == "v2"
        assert trace.attributes["cache_validation_stored"] == "v1"
        assert trace.attributes["cache_hit"] is False
        assert trace.attributes["cache_status"] == "invalidated"
        assert trace.attributes["cache_written"] is True
    finally:
        await memory_store.delete(key)
