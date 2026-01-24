"""Wrapper decorator with an LLM call.

Prints a message before and after the LLM call using a custom Wrapper.
The wrapper must return an async context manager.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

import litellm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, Result, Tree

# Requirements:
#   - OPENROUTER_API_KEY set for OpenRouter access via LiteLLM


@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def write_response(b: Blackboard, data: str) -> None:
    b.response = data


@asynccontextmanager
async def around_llm(child, context) -> AsyncGenerator[Result, None]:
    print("[wrapper] before LLM call")
    try:
        result = await child(context)
        yield result
    finally:
        print("[wrapper] after LLM call")


# fmt: off
tree = (
    Tree[Blackboard]("WrapperLLM")
    .Sequence()
    ._().Wrapper(around_llm)
    ._()._().LLM("openrouter/openai/gpt-4.1-mini", make_messages)
    ._().WriteBlackboard(write_response)
    .End()
)
# fmt: on


async def main() -> None:
    blackboard = Blackboard(prompt="Write a short haiku about clouds.")
    context = Context()

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Response:", blackboard.response)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")

    # prevent litellm introducing wired at-exit warnings..

    await litellm.close_litellm_async_clients()


if __name__ == "__main__":
    asyncio.run(main())
