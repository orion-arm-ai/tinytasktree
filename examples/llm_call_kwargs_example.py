"""LLM call with extra kwargs forwarded to LiteLLM.

Shows how to pass arbitrary keyword arguments (e.g. temperature, max_tokens)
through Tree.LLM via **llm_call_kwargs.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/" + "..")  # ensures tinytasktree is in module path

import asyncio
from dataclasses import dataclass

from tinytasktree import Context, FileTraceStorageHandler, Tree

# Running this example requires setting OS ENV variable `OPENROUTER_API_KEY`.


@dataclass
class Blackboard:
    prompt: str
    response: str = ""


def write_response(b: Blackboard, data: str) -> None:
    b.response = data


def make_messages(b: Blackboard) -> list[dict]:
    return [{"role": "user", "content": b.prompt}]


# fmt: off
tree = (
    Tree[Blackboard]("LLMCallKwargs")
    .Sequence()
    ._().LLM("openrouter/openai/gpt-4.1-mini", make_messages, temperature=0.2, max_tokens=128, top_p=0.9)
    ._().WriteBlackboard(write_response)
    .End()
)
# fmt: on


async def main() -> None:
    context = Context()
    blackboard = Blackboard(prompt="Summarize the benefits of unit testing in 3 sentences.")

    async with context.using_blackboard(blackboard):
        result = await tree(context)

    print("Result:", result)
    print("Response:", blackboard.response)

    storage = FileTraceStorageHandler(".traces")
    trace_id = await storage.save(context.trace_root())
    print("Trace URL:", f"http://127.0.0.1:5173/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())
