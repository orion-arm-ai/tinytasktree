"""RedisCacher behavior tests.

Steps:
- Run a RedisCacher without a value_validator to observe miss then hit.
- Run a RedisCacher with a value_validator to observe hit when validator matches and miss when it changes.
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


async def test_redis_cacher_without_value_validator_hit_and_miss(redis_client):
    key = f"test:redis-cacher:{uuid.uuid4()}"
    await redis_client.delete(key)

    try:
        calls: list[str] = []

        def compute_counted(b: Blackboard) -> int:
            calls.append("call")
            return compute(b)

        # fmt: off
        tree = (
            tinytasktree.Tree[Blackboard]("RedisCacherNoValidator")
            .RedisCacher(key_func=lambda b: b.key, redis_client=redis_client, expiration=5)
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

        context = tinytasktree.Context()
        blackboard = Blackboard(key=key, value=0)
        async with context.using_blackboard(blackboard):
            result = await tree(context)
        assert result.is_ok()
        assert result.data == 1
        assert blackboard.value == 0
        assert len(calls) == 1
    finally:
        await redis_client.delete(key)


async def test_redis_cacher_with_value_validator_hit_and_miss(redis_client):
    key = f"test:redis-cacher:{uuid.uuid4()}"
    await redis_client.delete(key)

    try:
        calls: list[str] = []

        def compute_counted(b: Blackboard) -> int:
            calls.append("call")
            return compute(b)

        # fmt: off
        tree = (
            tinytasktree.Tree[Blackboard]("RedisCacherWithValidator")
            .RedisCacher(
                key_func=lambda b: b.key,
                redis_client=redis_client,
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

        context = tinytasktree.Context()
        blackboard = Blackboard(key=key, value=0, validator="v1")
        async with context.using_blackboard(blackboard):
            result = await tree(context)
        assert result.is_ok()
        assert result.data == 1
        assert blackboard.value == 0
        assert len(calls) == 1

        context = tinytasktree.Context()
        blackboard = Blackboard(key=key, value=0, validator="v2")
        async with context.using_blackboard(blackboard):
            result = await tree(context)
        assert result.is_ok()
        assert result.data == 1
        assert blackboard.value == 1
        assert len(calls) == 2
    finally:
        await redis_client.delete(key)
