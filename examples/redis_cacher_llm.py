"""RedisCacher example with an LLM call.

Calls an LLM once and caches the result in Redis. A second call with the same
prompt hits the cache and returns much faster.
"""

import asyncio
import os
import random
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, Tree, set_default_global_redis_client

# Requirements:
#   - OPENROUTER_API_KEY set for OpenRouter access via LiteLLM
#   - Redis running and REDIS_URL set (default: redis://127.0.0.1:6379)


@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def cache_key(b: Blackboard) -> str:
    return f"tinytasktree:example:cache:{b.prompt}"


set_default_global_redis_client(url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379"))

# fmt: off
tree = (
    Tree[Blackboard]("RedisCacherLLM")
    .RedisCacher(key_func=cache_key, expiration=60)
    ._().Sequence()
    ._()._().LLM("openrouter/openai/gpt-4.1-mini", make_messages)
    ._()._().WriteBlackboard("response")
    .End()
)
# fmt: on

storage = FileTraceStorageHandler(".traces")


async def run_once(prompt: str) -> tuple[float, str]:
    context = Context()
    blackboard = Blackboard(prompt=prompt)
    start = time.monotonic()
    async with context.using_blackboard(blackboard):
        await tree(context)
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")
    duration = time.monotonic() - start
    return duration, blackboard.response


async def main() -> None:
    a = random.randint(1, 10)
    b = random.randint(1, 10)

    prompt = f"What is {a} * {b}? Answer with just the number."
    print(f"Prompt: {prompt}")

    dur1, resp1 = await run_once(prompt)
    print(f"First call:  {dur1:.3f}s -> {resp1}")

    dur2, resp2 = await run_once(prompt)
    print(f"Second call: {dur2:.3f}s -> {resp2}")


if __name__ == "__main__":
    asyncio.run(main())
