"""Cacher example with an LLM call.

Calls an LLM once and caches the result in a store. A second call with the same
prompt hits the cache and returns much faster.
"""

import asyncio
import os
import random
import sys
import time
from dataclasses import dataclass

import redis.asyncio as async_redis

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, LLMModel, LLMProvider, Tree

# Requirements:
#   - LLM_BASE_URL and LLM_API_KEY set for your LLM service
#   - redis-py installed and Redis running, with REDIS_URL set (default: redis://127.0.0.1:6379)
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
PROVIDER = LLMProvider(base_url=LLM_BASE_URL or "", api_key=LLM_API_KEY)
qwen3p6_plus = LLMModel("qwen/qwen3.6-plus", provider=PROVIDER, llm_call_kwargs={"reasoning": {"enabled": False}})


@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def cache_key(b: Blackboard) -> str:
    return f"tinytasktree:example:cache:{b.prompt}"


store = async_redis.Redis.from_url(os.getenv("REDIS_URL", "redis://127.0.0.1:6379"))

# fmt: off
tree = (
    Tree[Blackboard]("CacherLLM")
    .Cacher(key_func=cache_key, store=store, expiration=60)
    ._().Sequence()
    ._()._().LLM(qwen3p6_plus, make_messages)
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
    print("Trace URL:", f"http://127.0.0.1:8000/{trace_id}")
    duration = time.monotonic() - start
    return duration, blackboard.response


async def main() -> None:
    try:
        a = random.randint(1, 10)
        b = random.randint(1, 10)

        prompt = f"What is {a} * {b}? Answer with just the number."
        print(f"Prompt: {prompt}")

        dur1, resp1 = await run_once(prompt)
        print(f"First call:  {dur1:.3f}s -> {resp1}")

        dur2, resp2 = await run_once(prompt)
        print(f"Second call: {dur2:.3f}s -> {resp2}")
    finally:
        await store.aclose()


if __name__ == "__main__":
    asyncio.run(main())
