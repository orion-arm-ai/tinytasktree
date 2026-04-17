"""Timeout decorator with an LLM call.

Runs an LLM call with just 2-seconds timeout. On timeout, the fallback branch returns
an explicit message so the tree can continue and print a result.
"""

import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensure tinytasktree is importable

from tinytasktree import JSON, Context, FileTraceStorageHandler, Result, Tree

# Requirements:
#   - OPENROUTER_BASE_URL and OPENROUTER_API_KEY set for OpenRouter access
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def on_timeout(b: Blackboard) -> Result:
    return Result.OK("[timed out]")


def write_response(b: Blackboard, data: str) -> None:
    b.response = data


# fmt: off
tree = (
    Tree[Blackboard]("TimeoutLLM")
    .Sequence()
    ._().Timeout(2)
    ._()._().LLM("openai/gpt-4.1-mini", make_messages, base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    ._()._().Fallback()
    ._()._()._().Function(on_timeout)
    ._().WriteBlackboard(write_response)
    .End()
)
# fmt: on


async def main() -> None:
    blackboard = Blackboard(prompt="Write a 200-word story about a turtle.")
    context = Context()

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Response:", blackboard.response)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")

if __name__ == "__main__":
    asyncio.run(main())
