"""Retry node behavior tests.

Steps:
- Retry a child that succeeds after a few failures.
- Retry a child that always fails until max_tries is reached.
- Retry with sleep schedule to ensure retries proceed.
Expectations:
- Successful retry returns OK with child data.
- Exhausted retries return FAIL(None).
- Retry count matches max_tries when failures persist.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    attempts: int = 0


def fail_then_succeed(b: Blackboard) -> tinytasktree.Result:
    b.attempts += 1
    if b.attempts < 3:
        return tinytasktree.Result.FAIL(None)
    return tinytasktree.Result.OK("ok")


def always_fail(b: Blackboard) -> tinytasktree.Result:
    b.attempts += 1
    return tinytasktree.Result.FAIL(None)


async def async_always_fail(b: Blackboard) -> tinytasktree.Result:
    b.attempts += 1
    return tinytasktree.Result.FAIL(None)


async def test_retry_succeeds_after_failures():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("RetrySuccess")
        .Retry(max_tries=3)
        ._().Function(fail_then_succeed)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "ok"
    assert blackboard.attempts == 3


async def test_retry_exhausted_returns_fail():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("RetryFail")
        .Retry(max_tries=3)
        ._().Function(always_fail)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert result.data is None
    assert blackboard.attempts == 3


async def test_retry_with_sleep_schedule():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("RetrySleep")
        .Retry(max_tries=3, sleep_secs=[0.01, 0.01, 0.01])
        ._().Function(always_fail)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert result.data is None
    assert blackboard.attempts == 3


async def test_retry_async_child_with_short_sleep_list():
    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("RetryAsyncShortSleep")
        .Retry(max_tries=3, sleep_secs=[0.01])
        ._().Function(async_always_fail)
        .End()
    )
    # fmt: on

    context = tinytasktree.Context()
    blackboard = Blackboard()
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert not result.is_ok()
    assert result.data is None
    assert blackboard.attempts == 3
