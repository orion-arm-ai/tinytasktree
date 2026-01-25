"""LLM stream_on_delta callback tests.

Steps:
- Run streaming LLM with a sync stream_on_delta callback.
- Run streaming LLM with an async stream_on_delta callback.
Expectations:
- Callbacks receive deltas and final completion signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    prompt: str


def make_messages(b: Blackboard) -> list[tinytasktree.JSON]:
    return [{"role": "user", "content": b.prompt}]


async def test_llm_stream_on_delta_sync(mock_litellm):
    seen: list[tuple[str, str, bool]] = []

    def on_delta(b: Blackboard, full: str, delta: str, finished: bool):
        seen.append((full, delta, finished))

    async def handler(**kwargs):
        async def gen():
            yield {
                "choices": [{"delta": {"content": "he"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            }
            yield {
                "choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
            }
        return gen()

    mock_litellm(handler=handler)

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("LLMStreamSync")
        .Sequence()
        ._().LLM("mock/stream", make_messages, stream=True, stream_on_delta=on_delta)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="hi")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "hello"
    assert seen[0] == ("he", "he", False)
    assert seen[1] == ("hello", "llo", False)
    assert seen[2] == ("hello", "", True)


async def test_llm_stream_on_delta_async(mock_litellm):
    seen: list[tuple[str, str, bool, str]] = []

    async def on_delta(b: Blackboard, full: str, delta: str, finished: bool, reason: str):
        seen.append((full, delta, finished, reason))

    async def handler(**kwargs):
        async def gen():
            yield {
                "choices": [{"delta": {"content": "he"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            }
            yield {
                "choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
            }
        return gen()

    mock_litellm(handler=handler)

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("LLMStreamAsync")
        .Sequence()
        ._().LLM("mock/stream", make_messages, stream=True, stream_on_delta=on_delta)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt="hi")
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "hello"
    assert seen[0] == ("he", "he", False, "")
    assert seen[1] == ("hello", "llo", False, "stop")
    assert seen[2] == ("hello", "", True, "stop")
