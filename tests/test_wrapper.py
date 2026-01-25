"""Wrapper node behavior tests.

Steps:
- Wrap a child with a valid async context manager that runs the child.
- Wrap a child with an invalid wrapper that is not an async context manager.
Expectations:
- Valid wrapper returns the wrapped result.
- Invalid wrapper results in FAIL(None) due to programming error.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    value: str = ""


async def test_wrapper_valid_async_context_manager():
    runtimes = 0

    @asynccontextmanager
    async def wrapper(node: tinytasktree.Node, context: tinytasktree.Context):
        result = await node(context)
        nonlocal runtimes
        runtimes += 1
        yield result
        runtimes += 1

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("WrapperOk")
        .Wrapper(wrapper)
        ._().Function(lambda: "ok")
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "ok"
    assert runtimes == 2


async def test_wrapper_invalid_async_context_manager():
    def bad_wrapper(node: tinytasktree.Node, context: tinytasktree.Context):
        return "not-a-context-manager"

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("WrapperBad")
        .Wrapper(bad_wrapper)
        ._().Function(lambda: "ok")
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert result.data is None
