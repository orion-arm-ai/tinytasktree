"""LLM node behavior tests.

Steps:
- Run non-stream LLM with a mocked response and verify output, tokens, and cost stats.
- Verify API key resolution from default factory and per-node override.
- Run stream LLM with a mocked async generator and verify output and token stats.
Expectations:
- LLM returns OK with expected content.
- Token and cost stats are recorded on the tracer.
- API key resolution passes through to the OpenAI client and tracer attributes are set.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import tinytasktree


@dataclass
class Blackboard:
    prompt: str
    base_url: str | None = None


def _find_first_trace_by_kind(root: tinytasktree.TraceNode, kind: str) -> tinytasktree.TraceNode:
    stack = [root]
    while stack:
        node = stack.pop()
        if node.kind == kind:
            return node
        stack.extend(node.children.values())
    raise AssertionError(f"trace node kind not found: {kind}")


def make_messages(b: Blackboard) -> list[tinytasktree.JSON]:
    return [{"role": "user", "content": b.prompt}]


async def test_llm_basic_non_stream_stats(mock_openai):
    recorded = {}

    async def handler(**kwargs):
        recorded.update(kwargs)
        return {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            "_hidden_params": {"response_cost": 1.23},
        }

    mock_openai(handler=handler)

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("LLMBasic")
        .Sequence()
        ._().LLM("mock/basic", make_messages, stream=False)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="hi")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "hello"
    assert recorded["model"] == "mock/basic"
    assert recorded["messages"] == [{"role": "user", "content": "hi"}]
    assert recorded["stream"] is False
    assert recorded["client_kwargs"] == {}

    trace = _find_first_trace_by_kind(context.trace_root(), "LLM")
    assert trace.attributes["tokens"] == {"prompt": 4, "completion": 2, "total": 6}
    assert trace.attributes["prompt_tokens"] == 4
    assert trace.attributes["completion_tokens"] == 2
    assert trace.attributes["total_tokens"] == 6
    assert trace.cost == pytest.approx(1.23)


async def test_llm_api_key_resolution(mock_openai):
    recorded = {}

    async def handler(**kwargs):
        recorded.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    try:
        tinytasktree.set_default_llm_api_key_factory(lambda b: "default-key")

        # fmt: off
        tree_default = (
            tinytasktree.Tree[Blackboard]("LLMDefaultKey")
            .Sequence()
            ._().LLM("mock/key", make_messages)
            .End()
        )
        # fmt: on

        context = tinytasktree.Context()
        blackboard = Blackboard(prompt="hi")
        async with context.using_blackboard(blackboard):
            result = await tree_default(context)
        assert result.is_ok()
        assert recorded["client_kwargs"].get("api_key") == "default-key"
        trace = _find_first_trace_by_kind(context.trace_root(), "LLM")
        assert trace.attributes["api_key"] == "***"

        recorded.clear()

        # fmt: off
        tree_node_key = (
            tinytasktree.Tree[Blackboard]("LLMNodeKey")
            .Sequence()
            ._().LLM("mock/key", make_messages, api_key="node-key")
            .End()
        )
        # fmt: on

        context = tinytasktree.Context()
        blackboard = Blackboard(prompt="hi")
        async with context.using_blackboard(blackboard):
            result = await tree_node_key(context)
        assert result.is_ok()
        assert recorded["client_kwargs"].get("api_key") == "node-key"
        trace = _find_first_trace_by_kind(context.trace_root(), "LLM")
        assert trace.attributes["api_key"] == "***"
    finally:
        tinytasktree.set_default_llm_api_key_factory(None)


async def test_llm_base_url_factory_keeps_model_independent(mock_openai):
    recorded = {}

    async def handler(**kwargs):
        recorded.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("LLMBaseURL")
        .Sequence()
        ._().LLM("openai/gpt-4.1-mini", make_messages, base_url=lambda b: b.base_url)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="hi", base_url="https://llm.example/v1")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert recorded["model"] == "openai/gpt-4.1-mini"
    assert recorded["client_kwargs"].get("base_url") == "https://llm.example/v1"

    trace = _find_first_trace_by_kind(context.trace_root(), "LLM")
    assert trace.attributes["base_url"] == "https://llm.example/v1"


async def test_llm_streaming_tokens(mock_openai):
    async def handler(**kwargs):
        assert kwargs.get("stream") is True

        async def gen():
            yield {
                "choices": [{"delta": {"content": "he"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
            }
            yield {
                "choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }

        return gen()

    mock_openai(handler=handler)

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("LLMStream")
        .Sequence()
        ._().LLM("mock/stream", make_messages, stream=True)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="hi")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "hello"

    trace = _find_first_trace_by_kind(context.trace_root(), "LLM")
    assert trace.attributes["tokens"] == {"prompt": 2, "completion": 3, "total": 5}
    assert trace.attributes["prompt_tokens"] == 2
    assert trace.attributes["completion_tokens"] == 3
    assert trace.attributes["total_tokens"] == 5


async def test_llm_reasoning_is_forwarded_via_extra_body(mock_openai):
    recorded = {}

    async def handler(**kwargs):
        recorded.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_openai(handler=handler)

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("LLMReasoning")
        .Sequence()
        ._().LLM(
            "qwen/qwen3.6-plus",
            make_messages,
            base_url=lambda b: b.base_url,
            reasoning={"enabled": False},
        )
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="hi", base_url="https://llm.example/v1")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert "reasoning" not in recorded
    assert recorded["extra_body"] == {"reasoning": {"enabled": False}}
