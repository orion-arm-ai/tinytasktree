"""Terminable node behavior tests.

Steps:
- Run a Terminable with a long-running child and trigger termination via Redis key.
- Run a Terminable without termination and allow the child to complete.
Expectations:
- Termination triggers the fallback child and returns its value.
- No termination returns the main child's value.
- Cancellation from termination does not leak as an exception.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

import tinytasktree


@dataclass
class Blackboard:
    job_id: str


def _key(b: Blackboard) -> str:
    return f"test:terminable:{b.job_id}"


async def test_terminable_with_termination(redis_client):
    async def long_task():
        await asyncio.sleep(0.5)
        return "done"

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("TerminableTerminated")
        .Terminable(_key, redis_client=redis_client, monitor_interval_ms=10)
        ._().Function(long_task)
        ._().Fallback()
        ._()._().Function(lambda: "fallback")
        .End()
    )
    # fmt: on

    job_id = str(uuid.uuid4())
    key = f"test:terminable:{job_id}"

    async def trigger_termination():
        await asyncio.sleep(0.05)
        await redis_client.set(key, "1")

    context = tinytasktree.Context()
    blackboard = Blackboard(job_id=job_id)
    async with context.using_blackboard(blackboard):
        run_task = asyncio.create_task(tree(context))
        await asyncio.sleep(0)
        trigger_task = asyncio.create_task(trigger_termination())
        result = await run_task
        await trigger_task

    assert result.is_ok()
    assert result.data == "fallback"


async def test_terminable_cancellation_does_not_leak(redis_client):
    async def long_task():
        await asyncio.sleep(0.5)
        return "done"

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("TerminableCancelSafe")
        .Terminable(_key, redis_client=redis_client, monitor_interval_ms=10)
        ._().Function(long_task)
        ._().Fallback()
        ._()._().Function(lambda: "fallback")
        .End()
    )
    # fmt: on

    job_id = str(uuid.uuid4())
    key = f"test:terminable:{job_id}"

    async def trigger_termination():
        await asyncio.sleep(0.05)
        await redis_client.set(key, "1")

    context = tinytasktree.Context()
    blackboard = Blackboard(job_id=job_id)
    async with context.using_blackboard(blackboard):
        run_task = asyncio.create_task(tree(context))
        await asyncio.sleep(0)
        trigger_task = asyncio.create_task(trigger_termination())
        result = await run_task
        await trigger_task

    assert result.is_ok()
    assert result.data == "fallback"


async def test_terminable_without_termination(redis_client):
    async def long_task():
        await asyncio.sleep(0.05)
        return "done"

    # fmt: off
    tree = (
        tinytasktree.Tree[Blackboard]("TerminableNoSignal")
        .Terminable(_key, redis_client=redis_client, monitor_interval_ms=10)
        ._().Function(long_task)
        .End()
    )
    # fmt: on

    job_id = str(uuid.uuid4())
    key = f"test:terminable:{job_id}"
    await redis_client.delete(key)

    context = tinytasktree.Context()
    blackboard = Blackboard(job_id=job_id)
    async with context.using_blackboard(blackboard):
        result = await tree(context)

    assert result.is_ok()
    assert result.data == "done"
