"""Terminable LLM with streaming and cancel signal.

Runs a streaming LLM node and lets you cancel it any time via Redis; on cancel,
the fallback node marks the response as cancelled.
"""

import asyncio
import contextlib
import os
import sys
import uuid
from dataclasses import dataclass

import redis.asyncio as async_redis

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, Result, Tree

# Requirements:
#   - OPENROUTER_API_KEY set for OpenRouter access via LiteLLM
#   - Redis running and REDIS_URL set (default: redis://127.0.0.1:6379)


@dataclass
class Blackboard:
    prompt: str
    job_id: str
    response: str = ""


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def on_delta(b: Blackboard, fulltext: str, delta: str, finished: bool) -> None:
    print(delta, end="", flush=True)


def write_response(b: Blackboard, data: str) -> None:
    b.response = data


def on_cancelled(b: Blackboard) -> Result:
    b.response = "[cancelled]"
    return Result.FAIL(None)


def cancel_key(b: Blackboard) -> str:
    return f"tinytasktree:example:terminable_llm:cancel:{b.job_id}"


async def wait_for_cancel(redis: async_redis.Redis, key: str) -> None:
    await asyncio.to_thread(input, "\nPress Enter to cancel...\n")
    await redis.set(key, "1")


redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
redis = async_redis.Redis.from_url(redis_url)

# fmt: off
tree = (
    Tree[Blackboard]("TerminableLLM")
    .Terminable(cancel_key, redis)
    ._().Sequence()
    ._()._().LLM("openrouter/google/gemma-3-27b-it:free", make_messages, stream=True, stream_on_delta=on_delta)
    ._()._().WriteBlackboard(write_response)
    ._().Fallback()
    ._()._().Function(on_cancelled)
    .End()
)
# fmt: on


async def main() -> None:
    blackboard = Blackboard(
        prompt="Write a story about a cat and a robot in around 100 words.",
        job_id=str(uuid.uuid4()),
    )

    context = Context()
    key = cancel_key(blackboard)

    async with context.using_blackboard(blackboard):
        cancel_task = asyncio.create_task(wait_for_cancel(redis, key))
        try:
            result = await tree(context)
        finally:
            cancel_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_task

    print("\nResult:", result)
    print("Response:", blackboard.response)

    await redis.aclose()

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())
