"""Parallel LLM execution tests.

Steps:
- Configure a mock LLM handler that blocks until both calls start.
- Build a Parallel tree with two LLM branches that parse JSON into the blackboard.
- Run the tree under a timeout guard.
Expectations:
- Both LLM calls start before either finishes (parallelism).
- Tree returns OK status and both parsed results are present.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    prompt_a: str
    prompt_b: str
    parsed_a: dict | None = None
    parsed_b: dict | None = None


def make_messages_a(b: Blackboard) -> list[tinytasktree.JSON]:
    return [{"role": "user", "content": b.prompt_a}]


def make_messages_b(b: Blackboard) -> list[tinytasktree.JSON]:
    return [{"role": "user", "content": b.prompt_b}]


async def test_parallel_llm_calls_start_together(mock_litellm):
    starts: list[float] = []
    ready = asyncio.Event()

    async def handler(**kwargs):
        starts.append(asyncio.get_running_loop().time())
        if len(starts) == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=0.5)

        model = kwargs.get("model")
        if model == "mock/a":
            content = '{"value": "a"}'
        elif model == "mock/b":
            content = '{"value": "b"}'
        else:
            content = '{"value": "unknown"}'
        return {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "_hidden_params": {"response_cost": 0.0},
        }

    mock_litellm(handler=handler)

    # fmt: off
    subtree_a = (
        tinytasktree.Tree[Blackboard]("LLMA")
        .Sequence()
        ._().LLM("mock/a", make_messages_a)
        ._().ParseJSON(dst="parsed_a")
        .End()
    )
    subtree_b = (
        tinytasktree.Tree[Blackboard]("LLMB")
        .Sequence()
        ._().LLM("mock/b", make_messages_b)
        ._().ParseJSON(dst="parsed_b")
        .End()
    )
    tree = (
        tinytasktree.Tree[Blackboard]("ParallelLLM")
        .Parallel()
        ._().Subtree(subtree_a)
        ._().Subtree(subtree_b)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard(prompt_a="A", prompt_b="B")
    async with context.using_blackboard(blackboard):
        result = await asyncio.wait_for(tree(context), timeout=1.0)

    assert result.is_ok()
    assert len(starts) == 2
    assert blackboard.parsed_a == {"value": "a"}
    assert blackboard.parsed_b == {"value": "b"}
