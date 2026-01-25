"""Timeout node behavior tests.

Steps:
- Run a Timeout with a fast child and ensure it completes.
- Run a Timeout with a slow child and no fallback.
- Run a Timeout with a slow child and a fallback child.
Expectations:
- Fast child returns OK with its value.
- Timeout without fallback returns FAIL(None).
- Timeout with fallback returns the fallback's value.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    value: str = ""


async def test_timeout_no_timeout():
    async def fast():
        await asyncio.sleep(0.05)
        return "fast"

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("TimeoutFast")
        .Timeout(1.0)
        ._().Function(fast)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "fast"


async def test_timeout_with_no_fallback():
    async def slow():
        await asyncio.sleep(0.5)
        return "slow"

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("TimeoutNoFallback")
        .Timeout(0.1)
        ._().Function(slow)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert result.data is None


async def test_timeout_with_fallback():
    async def slow():
        await asyncio.sleep(0.5)
        return "slow"

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("TimeoutFallback")
        .Timeout(0.1)
        ._().Function(slow)
        ._().Function(lambda: "fallback")
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "fallback"
