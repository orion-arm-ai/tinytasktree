"""Cacher with validation.

Caches an LLM response and uses a validation tag so cache entries are invalidated
when the prompt format changes.
"""

import asyncio
import os
import random
import sys
from dataclasses import dataclass

import redis.asyncio as async_redis

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, Tree

# Requirements:
#   - LLM_BASE_URL and LLM_API_KEY set for your LLM service
#   - redis-py installed and Redis running, with REDIS_URL set (default: redis://127.0.0.1:6379)
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

PROMPT_VERSION = "v1"


@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def cache_key(b: Blackboard) -> str:
    return f"tinytasktree:example:cache:{b.prompt}"


def cache_validator(b: Blackboard) -> str:
    return PROMPT_VERSION


store = async_redis.Redis.from_url(os.getenv("REDIS_URL", "redis://127.0.0.1:6379"))

# fmt: off
tree = (
    Tree[Blackboard]("CacherValidator")
    .Cacher(key_func=cache_key, store=store, expiration=60, value_validator=cache_validator)
    ._().Sequence()
    ._()._().LLM("qwen/qwen3.6-plus", make_messages, base_url=LLM_BASE_URL, api_key=LLM_API_KEY, reasoning={"enabled": False})
    ._()._().WriteBlackboard("response")
    .End()
)
# fmt: on

storage = FileTraceStorageHandler(".traces")


async def run_once(prompt: str) -> str:
    context = Context()
    blackboard = Blackboard(prompt=prompt)
    async with context.using_blackboard(blackboard):
        await tree(context)
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:8000/{trace_id}")
    return blackboard.response


async def main() -> None:
    try:
        a = random.randint(10, 99)
        b = random.randint(10, 99)

        prompt = f"What is {a} * {b}? Answer with just the number."
        print(f"Prompt: {prompt}")

        resp1 = await run_once(prompt)
        print(f"First call finished (expecting: cache miss): {resp1}")

        resp2 = await run_once(prompt)
        print(f"Second call finished (expecting: cache hit): {resp2}")

        global PROMPT_VERSION
        PROMPT_VERSION = "v2"  # invalidate cached value

        resp3 = await run_once(prompt)
        print(f"Third call finished (expecting: validator miss): {resp3}")
    finally:
        await store.aclose()


if __name__ == "__main__":
    asyncio.run(main())
