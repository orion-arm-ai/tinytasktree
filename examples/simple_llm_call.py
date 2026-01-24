"""Simple LLM call with streaming.

Builds a small tree that streams the LLM output to stdout and writes the full
response to the blackboard for later use.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensures tinytasktree is in module path

import asyncio
from dataclasses import dataclass

from tinytasktree import JSON, Context, FileTraceStorageHandler, Tree

# Running this example requires setting OS ENV variable `OPENROUTER_API_KEY`.


@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def write_response(b: Blackboard, data: str) -> None:
    b.response = data


def make_messages(b: Blackboard) -> list[JSON]:
    return [{"role": "user", "content": b.prompt}]


def on_delta(b: Blackboard, fulltext: str, delta: str, finished: bool) -> None:
    print(delta, end="")


# fmt: off
tree = (
    Tree[Blackboard]("HelloWorld")
    .Sequence()
    ._().LLM("openrouter/google/gemini-2.5-flash-lite", make_messages, stream=True, stream_on_delta=on_delta)
    ._().WriteBlackboard(write_response)
    .End()
)
# fmt: on


async def main() -> None:
    context = Context()
    blackboard = Blackboard(prompt="Write a poem of 300 words!")

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Full Response:", blackboard.response)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())
