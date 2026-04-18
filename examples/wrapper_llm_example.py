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

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, LLMModel, LLMProvider, Result, Tree

# Requirements:
#   - LLM_BASE_URL and LLM_API_KEY set for your LLM service
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
PROVIDER = LLMProvider(base_url=LLM_BASE_URL or "", api_key=LLM_API_KEY)
MODEL = LLMModel("qwen/qwen3.6-plus", provider=PROVIDER, extra_body={"reasoning": {"enabled": False}})


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
    ._()._().LLM(MODEL, make_messages)
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
    print("Trace URL:", f"http://127.0.0.1:8000/{trace_id}")

if __name__ == "__main__":
    asyncio.run(main())
